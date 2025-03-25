# Copyright 2025 the Regents of the Superior School of Computer Sciene (ESOM) IPN. All rights reserved.
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

import os
import argparse
import subprocess
import shutil

OUTPUTS_FOLDER = './outputs/'

METHOD = 'splatfacto'

WORKSPACE_FOLDER = './nerfstudio/'

EXPORTS_FOLDER_IN_WORKSPACE = './nerfstudio/exports/'

EXPORTS_FOLDER = './exports/splat/'

def main(object_name):

    output_object_path = OUTPUTS_FOLDER + object_name + '/' + METHOD + '/'

    # check if there is a saved object
    if not os.path.exists(output_object_path):
        raise AssertionError('There is a saved object')
    

    # Checking for last saved folder in object path

    last_folder = sorted(os.listdir(output_object_path))[-1]

    config_file_path = output_object_path + last_folder + '/' + 'config.yml'

    # ns-export gaussian-splat --load-config outputs/poster/splatfacto/2025-03-17_184747/config.yml --output-dir exports/splat


    export_command = [
        f'ns-export gaussian-splat --load-config {config_file_path} --output-dir exports/splat/{object_name}'
    ]

    for command in export_command:
        subprocess.run(command, shell=True)

    # copy the exported object to the workspace
    if not os.path.exists(EXPORTS_FOLDER_IN_WORKSPACE):
        os.makedirs(EXPORTS_FOLDER_IN_WORKSPACE)

    export_file = 'splat.ply'

    shutil.copy(EXPORTS_FOLDER + object_name + '/' + export_file, EXPORTS_FOLDER_IN_WORKSPACE + object_name + '.ply')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the room to a 3D model')
    parser.add_argument('--object_name', type=str, help='Name of the object to export')
    
    args = parser.parse_args()

    main(args.object_name)