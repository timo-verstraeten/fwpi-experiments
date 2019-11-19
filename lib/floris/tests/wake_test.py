"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from floris.wake import Wake
from .sample_inputs import SampleInputs


class WakeTest():
    def __init__(self):
        self.sample_inputs = SampleInputs()
        self.input_dict = self.build_input_dict()

    def build_input_dict(self):
        return self.sample_inputs.wake


def test_instantiation():
    test_class = WakeTest()
    assert Wake(test_class.input_dict) is not None
