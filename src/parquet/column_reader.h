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

#ifndef PARQUET_COLUMN_READER_H
#define PARQUET_COLUMN_READER_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "parquet/column_page.h"
#include "parquet/encoding.h"
#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"
#include "parquet/util/visibility.h"

namespace arrow {

class BitReader;
class RleDecoder;

}  // namespace arrow

namespace parquet {

class PARQUET_EXPORT LevelDecoder {
 public:
  LevelDecoder();
  ~LevelDecoder();

  // Initialize the LevelDecoder state with new data
  // and return the number of bytes consumed
  int SetData(Encoding::type encoding, int16_t max_level, int num_buffered_values,
              const uint8_t* data);

  // Decodes a batch of levels into an array and returns the number of levels decoded
  int Decode(int batch_size, int16_t* levels);

 private:
  int bit_width_;
  int num_values_remaining_;
  Encoding::type encoding_;
  std::unique_ptr<::arrow::RleDecoder> rle_decoder_;
  std::unique_ptr<::arrow::BitReader> bit_packed_decoder_;
};

class PARQUET_EXPORT ColumnReader {
 public:
  ColumnReader(const ColumnDescriptor*, std::unique_ptr<PageReader>,
               ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());
  virtual ~ColumnReader();

  static std::shared_ptr<ColumnReader> Make(
      const ColumnDescriptor* descr, std::unique_ptr<PageReader> pager,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  // Returns true if there are still values in this column.
  bool HasNext() {
    // Either there is no data page available yet, or the data page has been
    // exhausted
    if (num_buffered_values_ == 0 || num_decoded_values_ == num_buffered_values_) {
      if (!ReadNewPage() || num_buffered_values_ == 0) {
        return false;
      }
    }
    return true;
  }

  Type::type type() const { return descr_->physical_type(); }

  const ColumnDescriptor* descr() const { return descr_; }

 protected:
  virtual bool ReadNewPage() = 0;

  // Read multiple definition levels into preallocated memory
  //
  // Returns the number of decoded definition levels
  int64_t ReadDefinitionLevels(int64_t batch_size, int16_t* levels);

  // Read multiple repetition levels into preallocated memory
  // Returns the number of decoded repetition levels
  int64_t ReadRepetitionLevels(int64_t batch_size, int16_t* levels);

  const ColumnDescriptor* descr_;

  std::unique_ptr<PageReader> pager_;
  std::shared_ptr<Page> current_page_;

  // Not set if full schema for this field has no optional or repeated elements
  LevelDecoder definition_level_decoder_;

  // Not set for flat schemas.
  LevelDecoder repetition_level_decoder_;

  // The total number of values stored in the data page. This is the maximum of
  // the number of encoded definition levels or encoded values. For
  // non-repeated, required columns, this is equal to the number of encoded
  // values. For repeated or optional values, there may be fewer data values
  // than levels, and this tells you how many encoded levels there are in that
  // case.
  int64_t num_buffered_values_;

  // The number of values from the current data page that have been decoded
  // into memory
  int64_t num_decoded_values_;

  ::arrow::MemoryPool* pool_;
};

// API to read values from a single column. This is a main client facing API.
template <typename DType>
class PARQUET_EXPORT TypedColumnReader : public ColumnReader {
 public:
  typedef typename DType::c_type T;

  TypedColumnReader(const ColumnDescriptor* schema, std::unique_ptr<PageReader> pager,
                    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : ColumnReader(schema, std::move(pager), pool), current_decoder_(NULL) {}
  virtual ~TypedColumnReader() {}

  // Read a batch of repetition levels, definition levels, and values from the
  // column.
  //
  // Since null values are not stored in the values, the number of values read
  // may be less than the number of repetition and definition levels. With
  // nested data this is almost certainly true.
  //
  // Set def_levels or rep_levels to nullptr if you want to skip reading them.
  // This is only safe if you know through some other source that there are no
  // undefined values.
  //
  // To fully exhaust a row group, you must read batches until the number of
  // values read reaches the number of stored values according to the metadata.
  //
  // This API is the same for both V1 and V2 of the DataPage
  //
  // @returns: actual number of levels read (see values_read for number of values read)
  int64_t ReadBatch(int64_t batch_size, int16_t* def_levels, int16_t* rep_levels,
                    T* values, int64_t* values_read);

  /// Read a batch of repetition levels, definition levels, and values from the
  /// column and leave spaces for null entries on the lowest level in the values
  /// buffer.
  ///
  /// In comparision to ReadBatch the length of repetition and definition levels
  /// is the same as of the number of values read for max_definition_level == 1.
  /// In the case of max_definition_level > 1, the repetition and definition
  /// levels are larger than the values but the values include the null entries
  /// with definition_level == (max_definition_level - 1).
  ///
  /// To fully exhaust a row group, you must read batches until the number of
  /// values read reaches the number of stored values according to the metadata.
  ///
  /// @param batch_size the number of levels to read
  /// @param[out] def_levels The Parquet definition levels, output has
  ///   the length levels_read.
  /// @param[out] rep_levels The Parquet repetition levels, output has
  ///   the length levels_read.
  /// @param[out] values The values in the lowest nested level including
  ///   spacing for nulls on the lowest levels; output has the length
  ///   values_read.
  /// @param[out] valid_bits Memory allocated for a bitmap that indicates if
  ///   the row is null or on the maximum definition level. For performance
  ///   reasons the underlying buffer should be able to store 1 bit more than
  ///   required. If this requires an additional byte, this byte is only read
  ///   but never written to.
  /// @param valid_bits_offset The offset in bits of the valid_bits where the
  ///   first relevant bit resides.
  /// @param[out] levels_read The number of repetition/definition levels that were read.
  /// @param[out] values_read The number of values read, this includes all
  ///   non-null entries as well as all null-entries on the lowest level
  ///   (i.e. definition_level == max_definition_level - 1)
  /// @param[out] null_count The number of nulls on the lowest levels.
  ///   (i.e. (values_read - null_count) is total number of non-null entries)
  int64_t ReadBatchSpaced(int64_t batch_size, int16_t* def_levels, int16_t* rep_levels,
                          T* values, uint8_t* valid_bits, int64_t valid_bits_offset,
                          int64_t* levels_read, int64_t* values_read,
                          int64_t* null_count);

  // Skip reading levels
  // Returns the number of levels skipped
  int64_t Skip(int64_t num_rows_to_skip);

 protected:
  typedef Decoder<DType> DecoderType;

  // Advance to the next data page
  virtual bool ReadNewPage();

  // Read up to batch_size values from the current data page into the
  // pre-allocated memory T*
  //
  // @returns: the number of values read into the out buffer
  int64_t ReadValues(int64_t batch_size, T* out);

  // Read up to batch_size values from the current data page into the
  // pre-allocated memory T*, leaving spaces for null entries according
  // to the def_levels.
  //
  // @returns: the number of values read into the out buffer
  int64_t ReadValuesSpaced(int64_t batch_size, T* out, int64_t null_count,
                           uint8_t* valid_bits, int64_t valid_bits_offset);

  // Map of encoding type to the respective decoder object. For example, a
  // column chunk's data pages may include both dictionary-encoded and
  // plain-encoded data.
  std::unordered_map<int, std::shared_ptr<DecoderType>> decoders_;

  void ConfigureDictionary(const DictionaryPage* page);

  DecoderType* current_decoder_;
};

/// \brief Stateful column reader that delimits semantic records for both flat
/// and nested columns
template <typename DType>
class PARQUET_EXPORT RecordReader : public TypedColumnReader<DType> {
  RecordReader(const ColumnDescriptor* descr, ::arrow::MemoryPool* pool);

  /// \return Number of records read
  int64_t ReadRecords(int64_t num_records);

  int64_t records_read() const { return records_read_; }

  int16_t* def_levels() { return reinterpret_cast<int16_t*>(def_levels_.mutable_data()); }

  int16_t* rep_levels() { return reinterpret_cast<int16_t*>(rep_levels_.mutable_data()); }

  T* values() { return reinterpret_cast<T*>(values_.mutable_data()); }

  /// \brief Number of values written including nulls (if any)
  int64_t values_written() const { return values_written_; }

  int64_t levels_written() const { return num_decoded_values_; }

  bool nullable_values() const { return nullable_values_; }

 private:
  void Reset();
  void Reserve(int64_t capacity);
  void AdvanceWritten(int64_t levels_written, int64_t values_written);
  void Advance(int64_t levels_consumed, int64_t values_consumed);

  // Process written repetition/definition levels to reach the end of
  // records. Process no more levels than necessary to delimit the indicated
  // number of logical records. Updates internal state of RecordReader
  //
  // \return Number of records delimited
  int64_t DelimitRecords(int64_t num_records, int64_t* values_seen);

  void PopulateValues(int64_t values_to_read);

  const int max_def_level_;
  const int max_rep_level_;

  bool nullable_values_;

  bool at_record_start_;
  int64_t records_read_;

  int64_t values_written_;
  int64_t values_capacity_;
  int64_t null_count_;

  int64_t levels_position_;
  int64_t levels_capacity_;

  // TODO(wesm): ByteArray / FixedLenByteArray types
  std::unique_ptr<::arrow::ArrayBuilder> builder_;

  std::shared_ptr<::arrow::PoolBuffer> values_;
  std::shared_ptr<::arrow::PoolBuffer> valid_bits_;
  std::shared_ptr<::arrow::PoolBuffer> def_levels_;
  std::shared_ptr<::arrow::PoolBuffer> rep_levels_;
};

typedef TypedColumnReader<BooleanType> BoolReader;
typedef TypedColumnReader<Int32Type> Int32Reader;
typedef TypedColumnReader<Int64Type> Int64Reader;
typedef TypedColumnReader<Int96Type> Int96Reader;
typedef TypedColumnReader<FloatType> FloatReader;
typedef TypedColumnReader<DoubleType> DoubleReader;
typedef TypedColumnReader<ByteArrayType> ByteArrayReader;
typedef TypedColumnReader<FLBAType> FixedLenByteArrayReader;

extern template class PARQUET_EXPORT TypedColumnReader<BooleanType>;
extern template class PARQUET_EXPORT TypedColumnReader<Int32Type>;
extern template class PARQUET_EXPORT TypedColumnReader<Int64Type>;
extern template class PARQUET_EXPORT TypedColumnReader<Int96Type>;
extern template class PARQUET_EXPORT TypedColumnReader<FloatType>;
extern template class PARQUET_EXPORT TypedColumnReader<DoubleType>;
extern template class PARQUET_EXPORT TypedColumnReader<ByteArrayType>;
extern template class PARQUET_EXPORT TypedColumnReader<FLBAType>;

extern template class PARQUET_EXPORT RecordReader<BooleanType>;
extern template class PARQUET_EXPORT RecordReader<Int32Type>;
extern template class PARQUET_EXPORT RecordReader<Int64Type>;
extern template class PARQUET_EXPORT RecordReader<Int96Type>;
extern template class PARQUET_EXPORT RecordReader<FloatType>;
extern template class PARQUET_EXPORT RecordReader<DoubleType>;
extern template class PARQUET_EXPORT RecordReader<ByteArrayType>;
extern template class PARQUET_EXPORT RecordReader<FLBAType>;

}  // namespace parquet

#endif  // PARQUET_COLUMN_READER_H
