# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
import os
import sys


IMPORT_REPLACEMENTS = [
    ('parquet/util/logging.h', 'arrow/util/logging.h'),
    ('parquet/util/visibility.h', 'parquet/util/macros.h')
]


MACRO_REPLACEMENTS = [
    ('PARQUET_PREDICT_FALSE', 'ARROW_PREDICT_FALSE'),
    ('PARQUET_PREDICT_TRUE', 'ARROW_PREDICT_TRUE'),
    ('PARQUET_NORETURN', 'ARROW_NORETURN'),
    ('PARQUET_DEPRECATED', 'ARROW_DEPRECATED'),
    ('PARQUET_TEMPLATE_EXPORT', 'ARROW_TEMPLATE_EXPORT')
]


def clean_file(path):
    with open(path) as f:
        lines = list(f)

    need_rewrite = False

    print("Fixing file: {0}".format(path))

    clean_lines = []
    for line in lines:
        clean_line = _clean_imports(line)
        clean_line = _clean_macros(clean_line)
        clean_lines.append(clean_line)
        if line != clean_line:
            print('OLD: {0}'.format(line))
            print('NEW: {0}'.format(clean_line))
            need_rewrite = True

    if need_rewrite:
        pass


def _clean_imports(line):
    for old_import, new_import in IMPORT_REPLACEMENTS:
        line = line.replace(old_import, new_import)
    return line


def _clean_macros(line):
    for old_macro, new_macro in MACRO_REPLACEMENTS:
        line = line.replace(old_macro, new_macro)
    return line


for dirpath, _, filenames in os.walk('src'):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)
        clean_file(full_path)
