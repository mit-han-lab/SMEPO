# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import traceback

from .utils import check_correctness as apps_check_correctness


def compute_score(completion, test_cases):
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)

        res, metadata = apps_check_correctness(
            in_outs=test_cases,
            generation=solution,
            timeout=5,
            debug=False,
        )

        try:
            metadata = dict(enumerate(metadata))[0]
        except Exception:
            metadata = {}
        passed = sum(1 for x in res if x is True)
        return passed == len(res), metadata

    except Exception:
        traceback.print_exc(10)
        return False, {}
