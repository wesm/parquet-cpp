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

#include "parquet/column_reader.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <arrow/util/bit-util.h>
#include <arrow/util/rle-encoding.h>

#include "parquet/column_page.h"
#include "parquet/encoding-internal.h"
#include "parquet/properties.h"

using arrow::MemoryPool;

namespace parquet {

LevelDecoder::LevelDecoder() : num_values_remaining_(0) {}

LevelDecoder::~LevelDecoder() {}

int LevelDecoder::SetData(Encoding::type encoding, int16_t max_level,
                          int num_buffered_values, const uint8_t* data) {
  int32_t num_bytes = 0;
  encoding_ = encoding;
  num_values_remaining_ = num_buffered_values;
  bit_width_ = BitUtil::Log2(max_level + 1);
  switch (encoding) {
    case Encoding::RLE: {
      num_bytes = *reinterpret_cast<const int32_t*>(data);
      const uint8_t* decoder_data = data + sizeof(int32_t);
      if (!rle_decoder_) {
        rle_decoder_.reset(new ::arrow::RleDecoder(decoder_data, num_bytes, bit_width_));
      } else {
        rle_decoder_->Reset(decoder_data, num_bytes, bit_width_);
      }
      return sizeof(int32_t) + num_bytes;
    }
    case Encoding::BIT_PACKED: {
      num_bytes =
          static_cast<int32_t>(BitUtil::Ceil(num_buffered_values * bit_width_, 8));
      if (!bit_packed_decoder_) {
        bit_packed_decoder_.reset(new ::arrow::BitReader(data, num_bytes));
      } else {
        bit_packed_decoder_->Reset(data, num_bytes);
      }
      return num_bytes;
    }
    default:
      throw ParquetException("Unknown encoding type for levels.");
  }
  return -1;
}

int LevelDecoder::Decode(int batch_size, int16_t* levels) {
  int num_decoded = 0;

  int num_values = std::min(num_values_remaining_, batch_size);
  if (encoding_ == Encoding::RLE) {
    num_decoded = rle_decoder_->GetBatch(levels, num_values);
  } else {
    num_decoded = bit_packed_decoder_->GetBatch(bit_width_, levels, num_values);
  }
  num_values_remaining_ -= num_decoded;
  return num_decoded;
}

ReaderProperties default_reader_properties() {
  static ReaderProperties default_reader_properties;
  return default_reader_properties;
}

ColumnReader::ColumnReader(const ColumnDescriptor* descr,
                           std::unique_ptr<PageReader> pager, MemoryPool* pool)
    : descr_(descr),
      pager_(std::move(pager)),
      num_buffered_values_(0),
      num_decoded_values_(0),
      pool_(pool) {}

ColumnReader::~ColumnReader() {}

template <typename DType>
void TypedColumnReader<DType>::ConfigureDictionary(const DictionaryPage* page) {
  int encoding = static_cast<int>(page->encoding());
  if (page->encoding() == Encoding::PLAIN_DICTIONARY ||
      page->encoding() == Encoding::PLAIN) {
    encoding = static_cast<int>(Encoding::RLE_DICTIONARY);
  }

  auto it = decoders_.find(encoding);
  if (it != decoders_.end()) {
    throw ParquetException("Column cannot have more than one dictionary.");
  }

  if (page->encoding() == Encoding::PLAIN_DICTIONARY ||
      page->encoding() == Encoding::PLAIN) {
    PlainDecoder<DType> dictionary(descr_);
    dictionary.SetData(page->num_values(), page->data(), page->size());

    // The dictionary is fully decoded during DictionaryDecoder::Init, so the
    // DictionaryPage buffer is no longer required after this step
    //
    // TODO(wesm): investigate whether this all-or-nothing decoding of the
    // dictionary makes sense and whether performance can be improved

    auto decoder = std::make_shared<DictionaryDecoder<DType>>(descr_, pool_);
    decoder->SetDict(&dictionary);
    decoders_[encoding] = decoder;
  } else {
    ParquetException::NYI("only plain dictionary encoding has been implemented");
  }

  current_decoder_ = decoders_[encoding].get();
}

// PLAIN_DICTIONARY is deprecated but used to be used as a dictionary index
// encoding.
static bool IsDictionaryIndexEncoding(const Encoding::type& e) {
  return e == Encoding::RLE_DICTIONARY || e == Encoding::PLAIN_DICTIONARY;
}

template <typename DType>
bool TypedColumnReader<DType>::ReadNewPage() {
  // Loop until we find the next data page.
  const uint8_t* buffer;

  while (true) {
    current_page_ = pager_->NextPage();
    if (!current_page_) {
      // EOS
      return false;
    }

    if (current_page_->type() == PageType::DICTIONARY_PAGE) {
      ConfigureDictionary(static_cast<const DictionaryPage*>(current_page_.get()));
      continue;
    } else if (current_page_->type() == PageType::DATA_PAGE) {
      const DataPage* page = static_cast<const DataPage*>(current_page_.get());

      // Read a data page.
      num_buffered_values_ = page->num_values();

      // Have not decoded any values from the data page yet
      num_decoded_values_ = 0;

      buffer = page->data();

      // If the data page includes repetition and definition levels, we
      // initialize the level decoder and subtract the encoded level bytes from
      // the page size to determine the number of bytes in the encoded data.
      int64_t data_size = page->size();

      // Data page Layout: Repetition Levels - Definition Levels - encoded values.
      // Levels are encoded as rle or bit-packed.
      // Init repetition levels
      if (descr_->max_repetition_level() > 0) {
        int64_t rep_levels_bytes = repetition_level_decoder_.SetData(
            page->repetition_level_encoding(), descr_->max_repetition_level(),
            static_cast<int>(num_buffered_values_), buffer);
        buffer += rep_levels_bytes;
        data_size -= rep_levels_bytes;
      }
      // TODO figure a way to set max_definition_level_ to 0
      // if the initial value is invalid

      // Init definition levels
      if (descr_->max_definition_level() > 0) {
        int64_t def_levels_bytes = definition_level_decoder_.SetData(
            page->definition_level_encoding(), descr_->max_definition_level(),
            static_cast<int>(num_buffered_values_), buffer);
        buffer += def_levels_bytes;
        data_size -= def_levels_bytes;
      }

      // Get a decoder object for this page or create a new decoder if this is the
      // first page with this encoding.
      Encoding::type encoding = page->encoding();

      if (IsDictionaryIndexEncoding(encoding)) {
        encoding = Encoding::RLE_DICTIONARY;
      }

      auto it = decoders_.find(static_cast<int>(encoding));
      if (it != decoders_.end()) {
        if (encoding == Encoding::RLE_DICTIONARY) {
          DCHECK(current_decoder_->encoding() == Encoding::RLE_DICTIONARY);
        }
        current_decoder_ = it->second.get();
      } else {
        switch (encoding) {
          case Encoding::PLAIN: {
            std::shared_ptr<DecoderType> decoder(new PlainDecoder<DType>(descr_));
            decoders_[static_cast<int>(encoding)] = decoder;
            current_decoder_ = decoder.get();
            break;
          }
          case Encoding::RLE_DICTIONARY:
            throw ParquetException("Dictionary page must be before data page.");

          case Encoding::DELTA_BINARY_PACKED:
          case Encoding::DELTA_LENGTH_BYTE_ARRAY:
          case Encoding::DELTA_BYTE_ARRAY:
            ParquetException::NYI("Unsupported encoding");

          default:
            throw ParquetException("Unknown encoding type.");
        }
      }
      current_decoder_->SetData(static_cast<int>(num_buffered_values_), buffer,
                                static_cast<int>(data_size));
      return true;
    } else {
      // We don't know what this page type is. We're allowed to skip non-data
      // pages.
      continue;
    }
  }
  return true;
}

// ----------------------------------------------------------------------
// Batch read APIs

int64_t ColumnReader::ReadDefinitionLevels(int64_t batch_size, int16_t* levels) {
  if (descr_->max_definition_level() == 0) {
    return 0;
  }
  return definition_level_decoder_.Decode(static_cast<int>(batch_size), levels);
}

int64_t ColumnReader::ReadRepetitionLevels(int64_t batch_size, int16_t* levels) {
  if (descr_->max_repetition_level() == 0) {
    return 0;
  }
  return repetition_level_decoder_.Decode(static_cast<int>(batch_size), levels);
}

// ----------------------------------------------------------------------
// Dynamic column reader constructor

std::shared_ptr<ColumnReader> ColumnReader::Make(const ColumnDescriptor* descr,
                                                 std::unique_ptr<PageReader> pager,
                                                 MemoryPool* pool) {
  switch (descr->physical_type()) {
    case Type::BOOLEAN:
      return std::make_shared<BoolReader>(descr, std::move(pager), pool);
    case Type::INT32:
      return std::make_shared<Int32Reader>(descr, std::move(pager), pool);
    case Type::INT64:
      return std::make_shared<Int64Reader>(descr, std::move(pager), pool);
    case Type::INT96:
      return std::make_shared<Int96Reader>(descr, std::move(pager), pool);
    case Type::FLOAT:
      return std::make_shared<FloatReader>(descr, std::move(pager), pool);
    case Type::DOUBLE:
      return std::make_shared<DoubleReader>(descr, std::move(pager), pool);
    case Type::BYTE_ARRAY:
      return std::make_shared<ByteArrayReader>(descr, std::move(pager), pool);
    case Type::FIXED_LEN_BYTE_ARRAY:
      return std::make_shared<FixedLenByteArrayReader>(descr, std::move(pager), pool);
    default:
      ParquetException::NYI("type reader not implemented");
  }
  // Unreachable code, but supress compiler warning
  return std::shared_ptr<ColumnReader>(nullptr);
}

namespace internal {

void DefinitionLevelsToBitmap(const int16_t* def_levels, int64_t num_def_levels,
                              int16_t max_definition_level, int16_t max_repetition_level,
                              int64_t* values_read, int64_t* null_count,
                              uint8_t* valid_bits, int64_t valid_bits_offset) {
  int64_t byte_offset = valid_bits_offset / 8;
  int64_t bit_offset = valid_bits_offset % 8;
  uint8_t bitset = valid_bits[byte_offset];

  // TODO(itaiin): As an interim solution we are splitting the code path here
  // between repeated+flat column reads, and non-repeated+nested reads.
  // Those paths need to be merged in the future
  for (int i = 0; i < num_def_levels; ++i) {
    if (def_levels[i] == max_definition_level) {
      bitset |= (1 << bit_offset);
    } else if (max_repetition_level > 0) {
      // repetition+flat case
      if (def_levels[i] == (max_definition_level - 1)) {
        bitset &= ~(1 << bit_offset);
        *null_count += 1;
      } else {
        continue;
      }
    } else {
      // non-repeated+nested case
      if (def_levels[i] < max_definition_level) {
        bitset &= ~(1 << bit_offset);
        *null_count += 1;
      } else {
        throw ParquetException("definition level exceeds maximum");
      }
    }

    bit_offset++;
    if (bit_offset == 8) {
      bit_offset = 0;
      valid_bits[byte_offset] = bitset;
      byte_offset++;
      // TODO: Except for the last byte, this shouldn't be needed
      bitset = valid_bits[byte_offset];
    }
  }
  if (bit_offset != 0) {
    valid_bits[byte_offset] = bitset;
  }
  *values_read = (bit_offset + byte_offset * 8 - valid_bits_offset);
}

}  // namespace internal

// ----------------------------------------------------------------------

RecordReader::RecordReader(const ColumnDescriptor* descr, MemoryPool* pool)
    : descr_(descr),
      pool_(pool),
      max_def_level_(descr->max_definition_level()),
      max_rep_level_(descr->max_repetition_level()),
      at_record_start_(false),
      records_read_(0),
      values_written_(0),
      values_capacity_(0),
      null_count_(0),
      levels_written_(0),
      levels_position_(0),
      levels_capacity_(0) {
  nullable_values_ = !descr->schema_node()->is_required();
  values_ = std::make_shared<PoolBuffer>(pool);
  valid_bits_ = std::make_shared<PoolBuffer>(pool);
  def_levels_ = std::make_shared<PoolBuffer>(pool);
  rep_levels_ = std::make_shared<PoolBuffer>(pool);

  if (descr->physical_type() == Type::BYTE_ARRAY) {
    builder_.reset(new ::arrow::BinaryBuilder(pool));
  } else if (descr->physical_type() == Type::FIXED_LEN_BYTE_ARRAY) {
    int byte_width = descr->type_length();
    std::shared_ptr<::arrow::DataType> type = ::arrow::fixed_size_binary(byte_width);
    builder_.reset(new ::arrow::FixedSizeBinaryBuilder(type, pool));
  }
  Reset();
}

int64_t RecordReader::ReadRecords(ColumnReader* reader, int64_t num_records) {
  // HasNext invokes ReadNewPage
  if (!reader->HasNext()) {
    return 0;
  }

  const int64_t start_levels_position = levels_position_;

  // Delimit records, then read values at the end
  int64_t values_seen = 0;
  int64_t values_to_read = 0;
  int64_t records_read = 0;

  if (levels_position_ < levels_written_) {
    records_read += DelimitRecords(num_records, &values_seen);
    values_to_read += values_seen;
  }

  DCHECK_EQ(levels_position_, levels_written_);

  // If we are in the middle of a record, we continue until reaching the
  // desired number of records or the end of the current record if we've found
  // enough records
  while (!at_record_start_ || records_read < num_records) {
    // Is there more data to read in this row group?
    if (!reader->HasNext()) {
      break;
    }

    /// We perform multiple batch reads until we either exhaust the row group
    /// or observe the desired number of records
    int64_t batch_size =
        std::min(num_records - records_read,
                 reader->buffered_values_current_page() - levels_written_);
    Reserve(batch_size);

    int16_t* def_levels = this->def_levels() + levels_written_;
    int16_t* rep_levels = this->rep_levels() + levels_written_;

    // Not present for non-repeated fields
    int64_t levels_read = 0;
    if (max_rep_level_ > 0) {
      levels_read = reader->ReadDefinitionLevels(batch_size, def_levels);
      if (reader->ReadRepetitionLevels(batch_size, rep_levels) != levels_read) {
        throw ParquetException("Number of decoded rep / def levels did not match");
      }
    } else if (max_def_level_ > 0) {
      levels_read = reader->ReadDefinitionLevels(batch_size, def_levels);
    }

    levels_written_ += levels_read;

    records_read += DelimitRecords(num_records - records_read, &values_seen);
    values_to_read += values_seen;

    // Exhausted data page
    if (levels_read == 0) {
      break;
    }
  }

  int64_t null_count = 0;
  if (nullable_values_) {
    int64_t implied_values_read = 0;
    uint8_t* valid_bits = valid_bits_->mutable_data();
    const int64_t valid_bits_offset = values_written_;

    internal::DefinitionLevelsToBitmap(
        this->def_levels() + start_levels_position,
        levels_position_ - start_levels_position, max_def_level_, max_rep_level_,
        &implied_values_read, &null_count, valid_bits, valid_bits_offset);
    ReadValuesSpaced(reader, values_to_read, null_count);
  } else {
    ReadValues(reader, values_to_read);
  }

  // Total values, including null spaces, if any
  values_written_ += values_to_read + null_count;
  null_count_ += null_count;

  return records_read;
}

int64_t RecordReader::DelimitRecords(int64_t num_records, int64_t* values_seen) {
  int64_t values_to_read = 0;
  int64_t records_read = 0;

  const int16_t* def_levels = this->def_levels();
  const int16_t* rep_levels = this->rep_levels();

  if (max_rep_level_ > 0) {
    // Count logical records and number of values to read
    for (; levels_position_ < levels_written_; ++levels_position_) {
      if (def_levels[levels_position_] == max_def_level_) {
        ++values_to_read;
      }
      if (rep_levels[levels_position_] == 0) {
        ++records_read;
        at_record_start_ = true;
      } else {
        at_record_start_ = false;
      }
    }
  } else {
    if (max_def_level_ > 0) {
      for (; levels_position_ < levels_written_; ++levels_position_) {
        if (def_levels[levels_position_] == max_def_level_) {
          ++values_to_read;
        }
        ++records_read;
      }
    } else {
      values_to_read = num_records;
      records_read = num_records;
    }
  }
  *values_seen = values_to_read;
  return records_read;
}

void RecordReader::ReadValues(ColumnReader* reader, int64_t values_to_read) {
  uint8_t* valid_bits = valid_bits_->mutable_data();
  const int64_t valid_bits_offset = values_written_;
  int64_t actual_read = 0;

#define READ_PRIMITIVE(PARQUET_TYPE)                                               \
  {                                                                                \
    auto typed_reader = static_cast<TypedColumnReader<PARQUET_TYPE>*>(reader);     \
    auto values =                                                                  \
        reinterpret_cast<typename PARQUET_TYPE::c_type*>(values_->mutable_data()); \
    actual_read = typed_reader->ReadValues(values_to_read, values);                \
    for (int64_t i = 0; i < values_to_read; i++) {                                 \
      ::arrow::BitUtil::SetBit(valid_bits, valid_bits_offset + i);                 \
    }                                                                              \
  }

  switch (descr_->physical_type()) {
    case Type::BOOLEAN:
      READ_PRIMITIVE(BooleanType);
      break;
    case Type::INT32:
      READ_PRIMITIVE(Int32Type);
      break;
    case Type::INT64:
      READ_PRIMITIVE(Int64Type);
      break;
    case Type::INT96:
      READ_PRIMITIVE(Int96Type);
      break;
    case Type::FLOAT:
      READ_PRIMITIVE(FloatType);
      break;
    case Type::DOUBLE:
      READ_PRIMITIVE(DoubleType);
      break;
    case Type::BYTE_ARRAY: {
      auto typed_reader = static_cast<ByteArrayReader*>(reader);
      auto values = reinterpret_cast<ByteArray*>(values_->mutable_data());
      actual_read = typed_reader->ReadValues(values_to_read, values);
      auto builder = static_cast<::arrow::BinaryBuilder*>(builder_.get());
      for (int64_t i = 0; i < actual_read; i++) {
        PARQUET_THROW_NOT_OK(
            builder->Append(values[i].ptr, static_cast<int64_t>(values[i].len)));
      }
      ResetValues();
    } break;
    case Type::FIXED_LEN_BYTE_ARRAY: {
      auto typed_reader = static_cast<FixedLenByteArrayReader*>(reader);
      auto values = reinterpret_cast<FLBA*>(values_->mutable_data());
      actual_read = typed_reader->ReadValues(values_to_read, values);
      auto builder = static_cast<::arrow::FixedSizeBinaryBuilder*>(builder_.get());
      for (int64_t i = 0; i < actual_read; i++) {
        PARQUET_THROW_NOT_OK(builder->Append(values[i].ptr));
      }
      ResetValues();
    } break;
  }

#undef READ_PRIMITIVE

  DCHECK_EQ(actual_read, values_to_read);
}

void RecordReader::ReadValuesSpaced(ColumnReader* reader, int64_t values_to_read,
                                    int64_t null_count) {
  uint8_t* valid_bits = valid_bits_->mutable_data();
  const int64_t valid_bits_offset = values_written_;
  int64_t actual_read = 0;

#define READ_PRIMITIVE(PARQUET_TYPE)                                                 \
  {                                                                                  \
    auto typed_reader = static_cast<TypedColumnReader<PARQUET_TYPE>*>(reader);       \
    auto values =                                                                    \
        reinterpret_cast<typename PARQUET_TYPE::c_type*>(values_->mutable_data());   \
    actual_read = typed_reader->ReadValuesSpaced(values_to_read, values, null_count, \
                                                 valid_bits, valid_bits_offset);     \
  }

  switch (descr_->physical_type()) {
    case Type::BOOLEAN:
      READ_PRIMITIVE(BooleanType);
      break;
    case Type::INT32:
      READ_PRIMITIVE(Int32Type);
      break;
    case Type::INT64:
      READ_PRIMITIVE(Int64Type);
      break;
    case Type::INT96:
      READ_PRIMITIVE(Int96Type);
      break;
    case Type::FLOAT:
      READ_PRIMITIVE(FloatType);
      break;
    case Type::DOUBLE:
      READ_PRIMITIVE(DoubleType);
      break;
    case Type::BYTE_ARRAY: {
      auto typed_reader = static_cast<ByteArrayReader*>(reader);
      auto values = reinterpret_cast<ByteArray*>(values_->mutable_data());
      actual_read = typed_reader->ReadValuesSpaced(values_to_read, values, null_count,
                                                   valid_bits, valid_bits_offset);
      auto builder = static_cast<::arrow::BinaryBuilder*>(builder_.get());

      for (int64_t i = 0; i < actual_read; i++) {
        if (::arrow::BitUtil::GetBit(valid_bits, valid_bits_offset + i)) {
          PARQUET_THROW_NOT_OK(
              builder->Append(values[i].ptr, static_cast<int64_t>(values[i].len)));
        } else {
          PARQUET_THROW_NOT_OK(builder->AppendNull());
        }
      }
      ResetValues();
    } break;
    case Type::FIXED_LEN_BYTE_ARRAY: {
      auto typed_reader = static_cast<FixedLenByteArrayReader*>(reader);
      auto values = reinterpret_cast<FLBA*>(values_->mutable_data());
      actual_read = typed_reader->ReadValuesSpaced(values_to_read, values, null_count,
                                                   valid_bits, valid_bits_offset);
      auto builder = static_cast<::arrow::FixedSizeBinaryBuilder*>(builder_.get());
      for (int64_t i = 0; i < actual_read; i++) {
        if (::arrow::BitUtil::GetBit(valid_bits, valid_bits_offset + i)) {
          PARQUET_THROW_NOT_OK(builder->Append(values[i].ptr));
        } else {
          PARQUET_THROW_NOT_OK(builder->AppendNull());
        }
      }
      ResetValues();
    } break;
  }
  DCHECK_EQ(actual_read, values_to_read);
}

void RecordReader::Reserve(int64_t capacity) {
  if (descr_->max_definition_level() > 0 &&
      (levels_written_ + capacity > levels_capacity_)) {
    int64_t new_levels_capacity = BitUtil::NextPower2(levels_capacity_ + 1);
    while (levels_written_ + capacity > new_levels_capacity) {
      new_levels_capacity = BitUtil::NextPower2(new_levels_capacity + 1);
    }
    PARQUET_THROW_NOT_OK(
        def_levels_->Resize(new_levels_capacity * sizeof(int16_t), false));

    if (nullable_values_) {
      PARQUET_THROW_NOT_OK(
          valid_bits_->Resize(BitUtil::BytesForBits(new_levels_capacity), false));
    }
    if (descr_->max_repetition_level() > 0) {
      PARQUET_THROW_NOT_OK(
          rep_levels_->Resize(new_levels_capacity * sizeof(int16_t), false));
    }
    levels_capacity_ = new_levels_capacity;
  }
  if (values_written_ + capacity > values_capacity_) {
    int64_t new_values_capacity = BitUtil::NextPower2(values_capacity_ + 1);
    while (values_written_ + capacity > new_values_capacity) {
      new_values_capacity = BitUtil::NextPower2(new_values_capacity + 1);
    }

    int type_size = GetTypeByteSize(descr_->physical_type());
    PARQUET_THROW_NOT_OK(values_->Resize(new_values_capacity * type_size, false));

    values_capacity_ = new_values_capacity;
  }
}

void RecordReader::Reset() {
  ResetValues();

  if (levels_written_ > 0) {
    const int64_t levels_remaining = levels_written_ - levels_position_;
    // Shift remaining levels to beginning of buffer and trim to only the number
    // of decoded levels remaining
    int16_t* def_data = def_levels();
    int16_t* rep_data = rep_levels();

    std::copy(def_data + levels_position_, def_data + levels_written_, def_data);
    std::copy(rep_data + levels_position_, rep_data + levels_written_, rep_data);

    PARQUET_THROW_NOT_OK(def_levels_->Resize(levels_remaining * sizeof(int16_t), false));
    PARQUET_THROW_NOT_OK(rep_levels_->Resize(levels_remaining * sizeof(int16_t), false));

    levels_written_ -= levels_position_;
    levels_position_ = 0;
    levels_capacity_ = levels_remaining;
  }

  records_read_ = 0;

  // Calling Finish on the builders also resets them
}

void RecordReader::ResetValues() {
  // Resize to 0, but do not shrink to fit
  PARQUET_THROW_NOT_OK(values_->Resize(0, false));
  PARQUET_THROW_NOT_OK(valid_bits_->Resize(0, false));
  values_written_ = 0;
  values_capacity_ = 0;
  null_count_ = 0;
}

// ----------------------------------------------------------------------
// Instantiate templated classes

template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<BooleanType>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<Int32Type>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<Int64Type>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<Int96Type>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<FloatType>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<DoubleType>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<ByteArrayType>;
template class PARQUET_TEMPLATE_EXPORT TypedColumnReader<FLBAType>;

}  // namespace parquet
