// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "parquet/arrow/reader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "arrow/api.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/logging.h"
#include "arrow/util/parallel.h"

#include "parquet/arrow/schema.h"
#include "parquet/schema.h"
#include "parquet/util/schema-util.h"

using arrow::Array;
using arrow::BooleanArray;
using arrow::Column;
using arrow::Field;
using arrow::Int32Array;
using arrow::ListArray;
using arrow::StructArray;
using arrow::MemoryPool;
using arrow::PoolBuffer;
using arrow::Status;
using arrow::Table;

using parquet::schema::NodePtr;

// Help reduce verbosity
using ParquetReader = parquet::ParquetFileReader;
using arrow::ParallelFor;

namespace parquet {
namespace arrow {

using ::arrow::BitUtil::BytesForBits;

constexpr int64_t kJulianToUnixEpochDays = 2440588LL;
constexpr int64_t kNanosecondsInADay = 86400LL * 1000LL * 1000LL * 1000LL;

static inline int64_t impala_timestamp_to_nanoseconds(const Int96& impala_timestamp) {
  int64_t days_since_epoch = impala_timestamp.value[2] - kJulianToUnixEpochDays;
  int64_t nanoseconds = *(reinterpret_cast<const int64_t*>(&(impala_timestamp.value)));
  return days_since_epoch * kNanosecondsInADay + nanoseconds;
}

template <typename ArrowType>
using ArrayType = typename ::arrow::TypeTraits<ArrowType>::ArrayType;

// ----------------------------------------------------------------------
// Iteration utilities

// Abstraction to decouple row group iteration details from the ColumnReader,
// so we can read only a single row group if we want
class FileColumnIterator {
 public:
  explicit FileColumnIterator(int column_index, ParquetFileReader* reader)
      : column_index_(column_index),
        reader_(reader),
        schema_(reader->metadata()->schema()) {}

  virtual ~FileColumnIterator() {}

  virtual std::shared_ptr<::parquet::ColumnReader> Next() = 0;

  const SchemaDescriptor* schema() const { return schema_; }

  const ColumnDescriptor* descr() const { return schema_->Column(column_index_); }

  std::shared_ptr<FileMetaData> metadata() const { return reader_->metadata(); }

  int column_index() const { return column_index_; }

 protected:
  int column_index_;
  ParquetFileReader* reader_;
  const SchemaDescriptor* schema_;
};

class AllRowGroupsIterator : public FileColumnIterator {
 public:
  explicit AllRowGroupsIterator(int column_index, ParquetFileReader* reader)
      : FileColumnIterator(column_index, reader), next_row_group_(0) {}

  std::shared_ptr<::parquet::ColumnReader> Next() override {
    std::shared_ptr<::parquet::ColumnReader> result;
    if (next_row_group_ < reader_->metadata()->num_row_groups()) {
      result = reader_->RowGroup(next_row_group_)->Column(column_index_);
      next_row_group_++;
    } else {
      result = nullptr;
    }
    return result;
  };

 private:
  int next_row_group_;
};

class SingleRowGroupIterator : public FileColumnIterator {
 public:
  explicit SingleRowGroupIterator(int column_index, int row_group_number,
                                  ParquetFileReader* reader)
      : FileColumnIterator(column_index, reader),
        row_group_number_(row_group_number),
        done_(false) {}

  std::shared_ptr<::parquet::ColumnReader> Next() override {
    if (done_) {
      return nullptr;
    }

    auto result = reader_->RowGroup(row_group_number_)->Column(column_index_);
    done_ = true;
    return result;
  };

 private:
  int row_group_number_;
  bool done_;
};

// ----------------------------------------------------------------------
// File reader implementation

class FileReader::Impl {
 public:
  Impl(MemoryPool* pool, std::unique_ptr<ParquetFileReader> reader)
      : pool_(pool), reader_(std::move(reader)), num_threads_(1) {}

  virtual ~Impl() {}

  Status GetColumn(int i, std::unique_ptr<ColumnReader>* out);
  Status ReadSchemaField(int i, std::shared_ptr<Array>* out);
  Status ReadSchemaField(int i, const std::vector<int>& indices,
                         std::shared_ptr<Array>* out);
  Status GetReaderForNode(int index, const NodePtr& node, const std::vector<int>& indices,
                          int16_t def_level, std::unique_ptr<ColumnReader::Impl>* out);
  Status ReadColumn(int i, std::shared_ptr<Array>* out);
  Status GetSchema(std::shared_ptr<::arrow::Schema>* out);
  Status GetSchema(const std::vector<int>& indices,
                   std::shared_ptr<::arrow::Schema>* out);
  Status ReadRowGroup(int row_group_index, const std::vector<int>& indices,
                      std::shared_ptr<::arrow::Table>* out);
  Status ReadTable(const std::vector<int>& indices, std::shared_ptr<Table>* table);
  Status ReadTable(std::shared_ptr<Table>* table);
  Status ReadRowGroup(int i, std::shared_ptr<Table>* table);

  bool CheckForFlatColumn(const ColumnDescriptor* descr);
  bool CheckForFlatListColumn(const ColumnDescriptor* descr);

  const ParquetFileReader* parquet_reader() const { return reader_.get(); }

  int num_row_groups() const { return reader_->metadata()->num_row_groups(); }

  void set_num_threads(int num_threads) { num_threads_ = num_threads; }

  ParquetFileReader* reader() { return reader_.get(); }

 private:
  MemoryPool* pool_;
  std::unique_ptr<ParquetFileReader> reader_;

  int num_threads_;
};

class ColumnReader::Impl {
 public:
  virtual ~Impl() {}
  virtual Status NextBatch(int batch_size, std::shared_ptr<Array>* out) = 0;
  virtual Status GetDefLevels(const int16_t** data, size_t* length) = 0;
  virtual Status GetRepLevels(const int16_t** data, size_t* length) = 0;
  virtual const std::shared_ptr<Field> field() = 0;
};

// Reader implementation for primitive arrays
class PARQUET_NO_EXPORT PrimitiveImpl : public ColumnReader::Impl {
 public:
  PrimitiveImpl(MemoryPool* pool, std::unique_ptr<FileColumnIterator> input)
      : pool_(pool), input_(std::move(input)), descr_(input_->descr()) {
    DCHECK(NodeToField(input_->descr()->schema_node(), &field_).ok());
    NextRowGroup();
  }

  virtual ~PrimitiveImpl() {}

  Status NextBatch(int records_to_read, std::shared_ptr<Array>* out) override;

  template <typename ArrowType, typename ParquetType>
  Status TypedReadBatch(int records_to_read, std::shared_ptr<Array>* out);

  template <typename ArrowType, typename ParquetType>
  Status WrapIntoListArray(std::shared_ptr<Array>* array);

  Status GetDefLevels(const int16_t** data, size_t* length) override;
  Status GetRepLevels(const int16_t** data, size_t* length) override;

  const std::shared_ptr<Field> field() override { return field_; }

 private:
  void NextRowGroup();

  MemoryPool* pool_;
  std::unique_ptr<FileColumnIterator> input_;
  const ColumnDescriptor* descr_;

  // An appropriate instance of a RecordReader<T>, which inherits from
  // ::parquet::ColumnReader
  std::shared_ptr<::parquet::ColumnReader> column_reader_;

  std::shared_ptr<Field> field_;
};

// Reader implementation for struct array
class PARQUET_NO_EXPORT StructImpl : public ColumnReader::Impl {
 public:
  explicit StructImpl(const std::vector<std::shared_ptr<Impl>>& children,
                      int16_t struct_def_level, MemoryPool* pool, const NodePtr& node)
      : children_(children),
        struct_def_level_(struct_def_level),
        pool_(pool),
        def_levels_buffer_(pool) {
    InitField(node, children);
  }

  virtual ~StructImpl() {}

  Status NextBatch(int batch_size, std::shared_ptr<Array>* out) override;
  Status GetDefLevels(const int16_t** data, size_t* length) override;
  Status GetRepLevels(const int16_t** data, size_t* length) override;
  const std::shared_ptr<Field> field() override { return field_; }

 private:
  std::vector<std::shared_ptr<Impl>> children_;
  int16_t struct_def_level_;
  MemoryPool* pool_;
  std::shared_ptr<Field> field_;
  PoolBuffer def_levels_buffer_;

  Status DefLevelsToNullArray(std::shared_ptr<MutableBuffer>* null_bitmap,
                              int64_t* null_count);
  void InitField(const NodePtr& node, const std::vector<std::shared_ptr<Impl>>& children);
};

FileReader::FileReader(MemoryPool* pool, std::unique_ptr<ParquetFileReader> reader)
    : impl_(new FileReader::Impl(pool, std::move(reader))) {}

FileReader::~FileReader() {}

Status FileReader::Impl::GetColumn(int i, std::unique_ptr<ColumnReader>* out) {
  std::unique_ptr<FileColumnIterator> input(new AllRowGroupsIterator(i, reader_.get()));

  std::unique_ptr<ColumnReader::Impl> impl(new PrimitiveImpl(pool_, std::move(input)));
  *out = std::unique_ptr<ColumnReader>(new ColumnReader(std::move(impl)));
  return Status::OK();
}

Status FileReader::Impl::GetReaderForNode(int index, const NodePtr& node,
                                          const std::vector<int>& indices,
                                          int16_t def_level,
                                          std::unique_ptr<ColumnReader::Impl>* out) {
  *out = nullptr;

  if (IsSimpleStruct(node)) {
    const schema::GroupNode* group = static_cast<const schema::GroupNode*>(node.get());
    std::vector<std::shared_ptr<ColumnReader::Impl>> children;
    for (int i = 0; i < group->field_count(); i++) {
      std::unique_ptr<ColumnReader::Impl> child_reader;
      // TODO(itaiin): Remove the -1 index hack when all types of nested reads
      // are supported. This currently just signals the lower level reader resolution
      // to abort
      RETURN_NOT_OK(GetReaderForNode(index, group->field(i), indices, def_level + 1,
                                     &child_reader));
      if (child_reader != nullptr) {
        children.push_back(std::move(child_reader));
      }
    }

    if (children.size() > 0) {
      *out = std::unique_ptr<ColumnReader::Impl>(
          new StructImpl(children, def_level, pool_, node));
    }
  } else {
    // This should be a flat field case - translate the field index to
    // the correct column index by walking down to the leaf node
    NodePtr walker = node;
    while (!walker->is_primitive()) {
      DCHECK(walker->is_group());
      auto group = static_cast<GroupNode*>(walker.get());
      if (group->field_count() != 1) {
        return Status::NotImplemented("lists with structs are not supported.");
      }
      walker = group->field(0);
    }
    auto column_index = reader_->metadata()->schema()->ColumnIndex(*walker.get());

    // If the index of the column is found then a reader for the coliumn is needed.
    // Otherwise *out keeps the nullptr value.
    if (std::find(indices.begin(), indices.end(), column_index) != indices.end()) {
      std::unique_ptr<ColumnReader> reader;
      RETURN_NOT_OK(GetColumn(column_index, &reader));
      *out = std::move(reader->impl_);
    }
  }

  return Status::OK();
}

Status FileReader::Impl::ReadSchemaField(int i, std::shared_ptr<Array>* out) {
  std::vector<int> indices(reader_->metadata()->num_columns());

  for (size_t j = 0; j < indices.size(); ++j) {
    indices[j] = static_cast<int>(j);
  }

  return ReadSchemaField(i, indices, out);
}

Status FileReader::Impl::ReadSchemaField(int i, const std::vector<int>& indices,
                                         std::shared_ptr<Array>* out) {
  auto parquet_schema = reader_->metadata()->schema();

  auto node = parquet_schema->group_node()->field(i);
  std::unique_ptr<ColumnReader::Impl> reader_impl;

  RETURN_NOT_OK(GetReaderForNode(i, node, indices, 1, &reader_impl));
  if (reader_impl == nullptr) {
    *out = nullptr;
    return Status::OK();
  }

  std::unique_ptr<ColumnReader> reader(new ColumnReader(std::move(reader_impl)));

  int64_t batch_size = 0;
  for (int j = 0; j < reader_->metadata()->num_row_groups(); j++) {
    batch_size += reader_->metadata()->RowGroup(j)->ColumnChunk(i)->num_values();
  }

  return reader->NextBatch(static_cast<int>(batch_size), out);
}

Status FileReader::Impl::ReadColumn(int i, std::shared_ptr<Array>* out) {
  std::unique_ptr<ColumnReader> flat_column_reader;
  RETURN_NOT_OK(GetColumn(i, &flat_column_reader));

  int64_t batch_size = 0;
  for (int j = 0; j < reader_->metadata()->num_row_groups(); j++) {
    batch_size += reader_->metadata()->RowGroup(j)->ColumnChunk(i)->num_values();
  }

  return flat_column_reader->NextBatch(static_cast<int>(batch_size), out);
}

Status FileReader::Impl::GetSchema(const std::vector<int>& indices,
                                   std::shared_ptr<::arrow::Schema>* out) {
  auto descr = reader_->metadata()->schema();

  std::cout << descr->ToString() << std::endl;

  auto parquet_key_value_metadata = reader_->metadata()->key_value_metadata();
  return FromParquetSchema(descr, indices, parquet_key_value_metadata, out);
}

Status FileReader::Impl::ReadRowGroup(int row_group_index,
                                      const std::vector<int>& indices,
                                      std::shared_ptr<::arrow::Table>* out) {
  std::shared_ptr<::arrow::Schema> schema;
  RETURN_NOT_OK(GetSchema(indices, &schema));

  auto rg_metadata = reader_->metadata()->RowGroup(row_group_index);

  int num_columns = static_cast<int>(indices.size());
  int nthreads = std::min<int>(num_threads_, num_columns);
  std::vector<std::shared_ptr<Column>> columns(num_columns);

  // TODO(wesm): Refactor to share more code with ReadTable

  auto ReadColumnFunc = [&indices, &row_group_index, &schema, &columns, &rg_metadata,
                         this](int i) {
    int column_index = indices[i];
    int64_t batch_size = rg_metadata->ColumnChunk(column_index)->num_values();

    std::unique_ptr<FileColumnIterator> input(
        new SingleRowGroupIterator(column_index, row_group_index, reader_.get()));

    std::unique_ptr<ColumnReader::Impl> impl(new PrimitiveImpl(pool_, std::move(input)));
    ColumnReader flat_column_reader(std::move(impl));

    std::shared_ptr<Array> array;
    RETURN_NOT_OK(flat_column_reader.NextBatch(static_cast<int>(batch_size), &array));
    columns[i] = std::make_shared<Column>(schema->field(i), array);
    return Status::OK();
  };

  if (nthreads == 1) {
    for (int i = 0; i < num_columns; i++) {
      RETURN_NOT_OK(ReadColumnFunc(i));
    }
  } else {
    RETURN_NOT_OK(ParallelFor(nthreads, num_columns, ReadColumnFunc));
  }

  *out = std::make_shared<Table>(schema, columns);
  return Status::OK();
}

Status FileReader::Impl::ReadTable(const std::vector<int>& indices,
                                   std::shared_ptr<Table>* table) {
  std::shared_ptr<::arrow::Schema> schema;
  RETURN_NOT_OK(GetSchema(indices, &schema));

  // We only need to read schema fields which have columns indicated
  // in the indices vector
  std::vector<int> field_indices;
  if (!ColumnIndicesToFieldIndices(*reader_->metadata()->schema(), indices,
                                   &field_indices)) {
    return Status::Invalid("Invalid column index");
  }

  std::vector<std::shared_ptr<Column>> columns(field_indices.size());
  auto ReadColumnFunc = [&indices, &field_indices, &schema, &columns, this](int i) {
    std::shared_ptr<Array> array;
    RETURN_NOT_OK(ReadSchemaField(field_indices[i], indices, &array));
    columns[i] = std::make_shared<Column>(schema->field(i), array);
    return Status::OK();
  };

  int num_fields = static_cast<int>(field_indices.size());
  int nthreads = std::min<int>(num_threads_, num_fields);
  if (nthreads == 1) {
    for (int i = 0; i < num_fields; i++) {
      RETURN_NOT_OK(ReadColumnFunc(i));
    }
  } else {
    RETURN_NOT_OK(ParallelFor(nthreads, num_fields, ReadColumnFunc));
  }

  *table = std::make_shared<Table>(schema, columns);
  return Status::OK();
}

Status FileReader::Impl::ReadTable(std::shared_ptr<Table>* table) {
  std::vector<int> indices(reader_->metadata()->num_columns());

  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
  }
  return ReadTable(indices, table);
}

Status FileReader::Impl::ReadRowGroup(int i, std::shared_ptr<Table>* table) {
  std::vector<int> indices(reader_->metadata()->num_columns());

  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = static_cast<int>(i);
  }
  return ReadRowGroup(i, indices, table);
}

// Static ctor
Status OpenFile(const std::shared_ptr<::arrow::io::ReadableFileInterface>& file,
                MemoryPool* allocator, const ReaderProperties& props,
                const std::shared_ptr<FileMetaData>& metadata,
                std::unique_ptr<FileReader>* reader) {
  std::unique_ptr<RandomAccessSource> io_wrapper(new ArrowInputFile(file));
  std::unique_ptr<ParquetReader> pq_reader;
  PARQUET_CATCH_NOT_OK(pq_reader =
                           ParquetReader::Open(std::move(io_wrapper), props, metadata));
  reader->reset(new FileReader(allocator, std::move(pq_reader)));
  return Status::OK();
}

Status OpenFile(const std::shared_ptr<::arrow::io::ReadableFileInterface>& file,
                MemoryPool* allocator, std::unique_ptr<FileReader>* reader) {
  return OpenFile(file, allocator, ::parquet::default_reader_properties(), nullptr,
                  reader);
}

Status FileReader::GetColumn(int i, std::unique_ptr<ColumnReader>* out) {
  return impl_->GetColumn(i, out);
}

Status FileReader::ReadColumn(int i, std::shared_ptr<Array>* out) {
  try {
    return impl_->ReadColumn(i, out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

Status FileReader::ReadSchemaField(int i, std::shared_ptr<Array>* out) {
  try {
    return impl_->ReadSchemaField(i, out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

Status FileReader::ReadTable(std::shared_ptr<Table>* out) {
  try {
    return impl_->ReadTable(out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

Status FileReader::ReadTable(const std::vector<int>& indices,
                             std::shared_ptr<Table>* out) {
  try {
    return impl_->ReadTable(indices, out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

Status FileReader::ReadRowGroup(int i, std::shared_ptr<Table>* out) {
  try {
    return impl_->ReadRowGroup(i, out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

Status FileReader::ReadRowGroup(int i, const std::vector<int>& indices,
                                std::shared_ptr<Table>* out) {
  try {
    return impl_->ReadRowGroup(i, indices, out);
  } catch (const ::parquet::ParquetException& e) {
    return ::arrow::Status::IOError(e.what());
  }
}

int FileReader::num_row_groups() const { return impl_->num_row_groups(); }

void FileReader::set_num_threads(int num_threads) { impl_->set_num_threads(num_threads); }

Status FileReader::ScanContents(std::vector<int> columns, const int32_t column_batch_size,
                                int64_t* num_rows) {
  try {
    *num_rows = ScanFileContents(columns, column_batch_size, impl_->reader());
    return Status::OK();
  } catch (const ::parquet::ParquetException& e) {
    return Status::IOError(e.what());
  }
}

const ParquetFileReader* FileReader::parquet_reader() const {
  return impl_->parquet_reader();
}

template <typename ArrowType, typename ParquetType>
Status PrimitiveImpl::WrapIntoListArray(std::shared_ptr<Array>* array) {
  auto reader = dynamic_cast<RecordReader<ParquetType>*>(column_reader_.get());
  const int16_t* def_levels = reader->def_levels();
  const int16_t* rep_levels = reader->rep_levels();
  const int64_t total_levels_read = reader->levels_position();

  std::shared_ptr<::arrow::Schema> arrow_schema;
  RETURN_NOT_OK(FromParquetSchema(input_->schema(), {input_->column_index()},
                                  input_->metadata()->key_value_metadata(),
                                  &arrow_schema));
  std::shared_ptr<Field> current_field = arrow_schema->field(0);

  if (descr_->max_repetition_level() > 0) {
    // Walk downwards to extract nullability
    std::vector<bool> nullable;
    std::vector<std::shared_ptr<::arrow::Int32Builder>> offset_builders;
    std::vector<std::shared_ptr<::arrow::BooleanBuilder>> valid_bits_builders;
    nullable.push_back(current_field->nullable());
    while (current_field->type()->num_children() > 0) {
      if (current_field->type()->num_children() > 1) {
        return Status::NotImplemented(
            "Fields with more than one child are not supported.");
      } else {
        if (current_field->type()->id() != ::arrow::Type::LIST) {
          return Status::NotImplemented(
              "Currently only nesting with Lists is supported.");
        }
        current_field = current_field->type()->child(0);
      }
      offset_builders.emplace_back(
          std::make_shared<::arrow::Int32Builder>(::arrow::int32(), pool_));
      valid_bits_builders.emplace_back(
          std::make_shared<::arrow::BooleanBuilder>(::arrow::boolean(), pool_));
      nullable.push_back(current_field->nullable());
    }

    int64_t list_depth = offset_builders.size();
    // This describes the minimal definition that describes a level that
    // reflects a value in the primitive values array.
    int16_t values_def_level = descr_->max_definition_level();
    if (nullable[nullable.size() - 1]) {
      values_def_level--;
    }

    // The definition levels that are needed so that a list is declared
    // as empty and not null.
    std::vector<int16_t> empty_def_level(list_depth);
    int def_level = 0;
    for (int i = 0; i < list_depth; i++) {
      if (nullable[i]) {
        def_level++;
      }
      empty_def_level[i] = def_level;
      def_level++;
    }

    int32_t values_offset = 0;
    std::vector<int64_t> null_counts(list_depth, 0);
    for (int64_t i = 0; i < total_levels_read; i++) {
      int16_t rep_level = rep_levels[i];
      if (rep_level < descr_->max_repetition_level()) {
        for (int64_t j = rep_level; j < list_depth; j++) {
          if (j == (list_depth - 1)) {
            RETURN_NOT_OK(offset_builders[j]->Append(values_offset));
          } else {
            RETURN_NOT_OK(offset_builders[j]->Append(
                static_cast<int32_t>(offset_builders[j + 1]->length())));
          }

          if (((empty_def_level[j] - 1) == def_levels[i]) && (nullable[j])) {
            RETURN_NOT_OK(valid_bits_builders[j]->Append(false));
            null_counts[j]++;
            break;
          } else {
            RETURN_NOT_OK(valid_bits_builders[j]->Append(true));
            if (empty_def_level[j] == def_levels[i]) {
              break;
            }
          }
        }
      }
      if (def_levels[i] >= values_def_level) {
        values_offset++;
      }
    }
    // Add the final offset to all lists
    for (int64_t j = 0; j < list_depth; j++) {
      if (j == (list_depth - 1)) {
        RETURN_NOT_OK(offset_builders[j]->Append(values_offset));
      } else {
        RETURN_NOT_OK(offset_builders[j]->Append(
            static_cast<int32_t>(offset_builders[j + 1]->length())));
      }
    }

    std::vector<std::shared_ptr<Buffer>> offsets;
    std::vector<std::shared_ptr<Buffer>> valid_bits;
    std::vector<int64_t> list_lengths;
    for (int64_t j = 0; j < list_depth; j++) {
      list_lengths.push_back(offset_builders[j]->length() - 1);
      std::shared_ptr<Array> array;
      RETURN_NOT_OK(offset_builders[j]->Finish(&array));
      offsets.emplace_back(std::static_pointer_cast<Int32Array>(array)->values());
      RETURN_NOT_OK(valid_bits_builders[j]->Finish(&array));
      valid_bits.emplace_back(std::static_pointer_cast<BooleanArray>(array)->values());
    }

    std::shared_ptr<Array> output(*array);
    for (int64_t j = list_depth - 1; j >= 0; j--) {
      auto list_type = std::make_shared<::arrow::ListType>(
          std::make_shared<Field>("item", output->type(), nullable[j + 1]));
      output = std::make_shared<::arrow::ListArray>(
          list_type, list_lengths[j], offsets[j], output, valid_bits[j], null_counts[j]);
    }
    *array = output;
  }
  return Status::OK();
}

template <typename ArrowType, typename ParquetType>
struct supports_fast_path_impl {
  using ArrowCType = typename ArrowType::c_type;
  using ParquetCType = typename ParquetType::c_type;
  static constexpr bool value = std::is_same<ArrowCType, ParquetCType>::value;
};

template <typename ArrowType, typename ParquetType>
using supports_fast_path =
    typename std::enable_if<supports_fast_path_impl<ArrowType, ParquetType>::value>::type;

template <typename ArrowType, typename ParquetType, typename Enable = void>
struct TransferFunctor {
  using ArrowCType = typename ArrowType::c_type;

  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->num_values();
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(::arrow::AllocateBuffer(pool, length * sizeof(ArrowCType), &data));

    auto out_ptr = reinterpret_cast<ArrowCType*>(data->mutable_data());
    std::copy(out_ptr, out_ptr + length, out_ptr);

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data,
                                                    is_valid, reader->null_count());
    } else {
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data);
    }
    return Status::OK();
  }
};

template <typename ArrowType, typename ParquetType>
struct TransferFunctor<ArrowType, ParquetType,
                       supports_fast_path<ArrowType, ParquetType>>{
  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->num_values();
    std::shared_ptr<PoolBuffer> values = reader->ReleaseValues();

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, values,
                                                    is_valid, reader->null_count());
    } else {
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, values);
    }
    return Status::OK();
  }
};

template <>
struct TransferFunctor<::arrow::BooleanType, BooleanType> {
  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->num_values();
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(::arrow::AllocateBuffer(pool, BytesForBits(length), &data));

    // Transfer boolean values to packed bitmap
    const bool* values = reader->values_written();
    uint8_t* data_ptr = data->mutable_data();
    for (int64_t i = 0; i < length; i++) {
      if (values[i]) {
        ::arrow::BitUtil::SetBit(data_ptr, i);
      }
    }

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      RETURN_NOT_OK(is_valid->Resize(BytesForBits(length), false));
      *out = std::make_shared<BooleanArray>(type, length, data, is_valid,
                                            reader->null_count());
    } else {
      *out = std::make_shared<BooleanArray>(type, length, data);
    }
    return Status::OK();
  }
};

template <>
struct TransferFunctor<::arrow::TimestampType, Int96Type> {
  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->values_written();
    Int96* values = reader->values;

    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(::arrow::AllocateBuffer(pool, length * sizeof(int64_t), &data));

    auto data_ptr = reinterpret_cast<int64_t*>(data->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      *data_ptr++ = impala_timestamp_to_nanoseconds(values[i]);
    }

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      *out = std::make_shared<TimestampArray>(type, length, data, is_valid,
                                              reader->null_count());
    } else {
      *out = std::make_shared<TimestampArray>(type, length, data);
    }

    return Status::OK();
  }
};

template <>
struct TransferFunctor<::arrow::Date64Type, Int32Type> {
  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->num_values();
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(::arrow::AllocateBuffer(pool, length * sizeof(int32_t), &data));

    auto out_ptr = reinterpret_cast<int32_t*>(data->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      *out_ptr++ = static_cast<int64_t>(values[i]) * 86400000;
    }

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data, is_valid,
                                                    reader->null_count());
    } else {
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data);
    }
    return Status::OK();
  }
};

template <typename ArrowType, typename ParquetType>
struct TransferFunctor<
  ArrowType, ParquetType,
  typename std::enable_if<std::is_same<ParquetType, ByteArrayType>::value ||
                          std::is_same<ParquetType, FLBAType>::value>::type>> {
  Status operator()(RecordReader<ParquetType>* reader,
                    MemoryPool* pool,
                    const std::shared_ptr<::arrow::DataType>& type,
                    std::shared_ptr<Array>* out) {
    int64_t length = reader->num_values();
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(::arrow::AllocateBuffer(pool, length * sizeof(int32_t), &data));

    auto out_ptr = reinterpret_cast<int32_t*>(data->mutable_data());
    for (int64_t i = 0; i < length; i++) {
      *out_ptr++ = static_cast<int64_t>(values[i]) * 86400000;
    }

    if (reader->nullable_values()) {
      std::shared_ptr<PoolBuffer> is_valid = reader->ReleaseIsValid();
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data, is_valid,
                                                    reader->null_count());
    } else {
      *out = std::make_shared<ArrayType<ArrowType>>(type, length, data);
    }
    return Status::OK();
  }
};

template <typename ArrowType, typename ParquetType>
Status PrimitiveImpl::TypedReadBatch(int64_t records_to_read,
                                     std::shared_ptr<Array>* out) {
  using ArrowCType = typename ArrowType::c_type;

  auto reader = dynamic_cast<RecordReader<ParquetType>*>(column_reader_.get());
  DCHECK(reader);

  PARQUET_CATCH_NOT_OK(reader->Reserve(batch_size));

  while ((records_to_read > 0) && column_reader_) {
    int64_t values_read;
    int64_t levels_read;
    records_to_read -= reader->ReadRecords(records_to_read);
    if (!column_reader_->HasNext()) {
      NextRowGroup();
    }
  }

  int64_t actual_records = std::min(records_to_read, reader->records_read());

  TransferFunctor<ArrowType, ParquetType> func;
  RETURN_NOT_OK(func(reader, pool_, field_->type(), out));

  // Check if we should transform this array into an list array.
  RETURN_NOT_OK(WrapIntoListArray(out));

  PARQUET_CATCH_NOT_OK(reader->Reset());

  return Status::OK();
}

#define TYPED_BATCH_CASE(ENUM, ArrowType, ParquetType)              \
  case ::arrow::Type::ENUM:                                         \
    return TypedReadBatch<ArrowType, ParquetType>(batch_size, out); \
    break;

Status PrimitiveImpl::NextBatch(int batch_size, std::shared_ptr<Array>* out) {
  if (!column_reader_) {
    // Exhausted all row groups.
    *out = nullptr;
    return Status::OK();
  }

  switch (field_->type()->id()) {
    case ::arrow::Type::NA:
      *out = std::make_shared<::arrow::NullArray>(batch_size);
      return Status::OK();
      break;
      TYPED_BATCH_CASE(BOOL, ::arrow::BooleanType, BooleanType)
      TYPED_BATCH_CASE(UINT8, ::arrow::UInt8Type, Int32Type)
      TYPED_BATCH_CASE(INT8, ::arrow::Int8Type, Int32Type)
      TYPED_BATCH_CASE(UINT16, ::arrow::UInt16Type, Int32Type)
      TYPED_BATCH_CASE(INT16, ::arrow::Int16Type, Int32Type)
      TYPED_BATCH_CASE(UINT32, ::arrow::UInt32Type, Int32Type)
      TYPED_BATCH_CASE(INT32, ::arrow::Int32Type, Int32Type)
      TYPED_BATCH_CASE(UINT64, ::arrow::UInt64Type, Int64Type)
      TYPED_BATCH_CASE(INT64, ::arrow::Int64Type, Int64Type)
      TYPED_BATCH_CASE(FLOAT, ::arrow::FloatType, FloatType)
      TYPED_BATCH_CASE(DOUBLE, ::arrow::DoubleType, DoubleType)
      TYPED_BATCH_CASE(STRING, ::arrow::StringType, ByteArrayType)
      TYPED_BATCH_CASE(BINARY, ::arrow::BinaryType, ByteArrayType)
      TYPED_BATCH_CASE(DATE32, ::arrow::Date32Type, Int32Type)
      TYPED_BATCH_CASE(DATE64, ::arrow::Date64Type, Int32Type)
      TYPED_BATCH_CASE(FIXED_SIZE_BINARY, ::arrow::FixedSizeBinaryType, FLBATypea)
    case ::arrow::Type::TIMESTAMP: {
      ::arrow::TimestampType* timestamp_type =
          static_cast<::arrow::TimestampType*>(field_->type().get());
      switch (timestamp_type->unit()) {
        case ::arrow::TimeUnit::MILLI:
          return TypedReadBatch<::arrow::TimestampType, Int64Type>(batch_size, out);
          break;
        case ::arrow::TimeUnit::MICRO:
          return TypedReadBatch<::arrow::TimestampType, Int64Type>(batch_size, out);
          break;
        case ::arrow::TimeUnit::NANO:
          return TypedReadBatch<::arrow::TimestampType, Int96Type>(batch_size, out);
          break;
        default:
          return Status::NotImplemented("TimeUnit not supported");
      }
      break;
    }
      TYPED_BATCH_CASE(TIME32, ::arrow::Time32Type, Int32Type)
      TYPED_BATCH_CASE(TIME64, ::arrow::Time64Type, Int64Type)
    default:
      std::stringstream ss;
      ss << "No support for reading columns of type " << field_->type()->ToString();
      return Status::NotImplemented(ss.str());
  }
}

void PrimitiveImpl::NextRowGroup() { column_reader_ = input_->Next(); }

Status PrimitiveImpl::GetDefLevels(const int16_t** data, size_t* length) {
  *data = reinterpret_cast<const int16_t*>(def_levels_buffer_.data());
  *length = def_levels_buffer_.size() / sizeof(int16_t);
  return Status::OK();
}

Status PrimitiveImpl::GetRepLevels(const int16_t** data, size_t* length) {
  *data = reinterpret_cast<const int16_t*>(rep_levels_buffer_.data());
  *length = rep_levels_buffer_.size() / sizeof(int16_t);
  return Status::OK();
}

ColumnReader::ColumnReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ColumnReader::~ColumnReader() {}

Status ColumnReader::NextBatch(int batch_size, std::shared_ptr<Array>* out) {
  return impl_->NextBatch(batch_size, out);
}

// StructImpl methods

Status StructImpl::DefLevelsToNullArray(std::shared_ptr<MutableBuffer>* null_bitmap_out,
                                        int64_t* null_count_out) {
  std::shared_ptr<MutableBuffer> null_bitmap;
  auto null_count = 0;
  const int16_t* def_levels_data;
  size_t def_levels_length;
  RETURN_NOT_OK(GetDefLevels(&def_levels_data, &def_levels_length));
  RETURN_NOT_OK(GetEmptyBitmap(pool_, def_levels_length, &null_bitmap));
  uint8_t* null_bitmap_ptr = null_bitmap->mutable_data();
  for (size_t i = 0; i < def_levels_length; i++) {
    if (def_levels_data[i] < struct_def_level_) {
      // Mark null
      null_count += 1;
    } else {
      DCHECK_EQ(def_levels_data[i], struct_def_level_);
      ::arrow::BitUtil::SetBit(null_bitmap_ptr, i);
    }
  }

  *null_count_out = null_count;
  *null_bitmap_out = (null_count == 0) ? nullptr : null_bitmap;
  return Status::OK();
}

// TODO(itaiin): Consider caching the results of this calculation -
//   note that this is only used once for each read for now
Status StructImpl::GetDefLevels(const int16_t** data, size_t* length) {
  *data = nullptr;
  if (children_.size() == 0) {
    // Empty struct
    *length = 0;
    return Status::OK();
  }

  // We have at least one child
  const int16_t* child_def_levels;
  size_t child_length;
  RETURN_NOT_OK(children_[0]->GetDefLevels(&child_def_levels, &child_length));
  auto size = child_length * sizeof(int16_t);
  RETURN_NOT_OK(def_levels_buffer_.Resize(size));
  // Initialize with the minimal def level
  std::memset(def_levels_buffer_.mutable_data(), -1, size);
  auto result_levels = reinterpret_cast<int16_t*>(def_levels_buffer_.mutable_data());

  // When a struct is defined, all of its children def levels are at least at
  // nesting level, and def level equals nesting level.
  // When a struct is not defined, all of its children def levels are less than
  // the nesting level, and the def level equals max(children def levels)
  // All other possibilities are malformed definition data.
  for (auto& child : children_) {
    size_t current_child_length;
    RETURN_NOT_OK(child->GetDefLevels(&child_def_levels, &current_child_length));
    DCHECK_EQ(child_length, current_child_length);
    for (size_t i = 0; i < child_length; i++) {
      // Check that value is either uninitialized, or current
      // and previous children def levels agree on the struct level
      DCHECK((result_levels[i] == -1) || ((result_levels[i] >= struct_def_level_) ==
                                          (child_def_levels[i] >= struct_def_level_)));
      result_levels[i] =
          std::max(result_levels[i], std::min(child_def_levels[i], struct_def_level_));
    }
  }
  *data = reinterpret_cast<const int16_t*>(def_levels_buffer_.data());
  *length = child_length;
  return Status::OK();
}

void StructImpl::InitField(const NodePtr& node,
                           const std::vector<std::shared_ptr<Impl>>& children) {
  // Make a shallow node to field conversion from the children fields
  std::vector<std::shared_ptr<::arrow::Field>> fields(children.size());
  for (size_t i = 0; i < children.size(); i++) {
    fields[i] = children[i]->field();
  }
  auto type = std::make_shared<::arrow::StructType>(fields);
  field_ = std::make_shared<Field>(node->name(), type);
}

Status StructImpl::GetRepLevels(const int16_t** data, size_t* length) {
  return Status::NotImplemented("GetRepLevels is not implemented for struct");
}

Status StructImpl::NextBatch(int batch_size, std::shared_ptr<Array>* out) {
  std::vector<std::shared_ptr<Array>> children_arrays;
  std::shared_ptr<MutableBuffer> null_bitmap;
  int64_t null_count;

  // Gather children arrays and def levels
  for (auto& child : children_) {
    std::shared_ptr<Array> child_array;

    RETURN_NOT_OK(child->NextBatch(batch_size, &child_array));

    children_arrays.push_back(child_array);
  }

  RETURN_NOT_OK(DefLevelsToNullArray(&null_bitmap, &null_count));

  *out = std::make_shared<StructArray>(field()->type(), batch_size, children_arrays,
                                       null_bitmap, null_count);

  return Status::OK();
}

}  // namespace arrow
}  // namespace parquet
