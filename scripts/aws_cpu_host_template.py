"""Generate the fixed-size Dockyard CPU host CloudFormation template."""

import argparse
import json
from pathlib import Path


def template():
    bootstrap = Path(__file__).with_name("setup_dockyard_cpu.sh").read_text()
    tags = [{"Key": "Project", "Value": "dockyard"}]
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "One On-Demand r6a.large with 20 GiB root and retained 80 GiB Dockyard data",
        "Parameters": {
            "AmiId": {"Type": "AWS::EC2::Image::Id"},
            "VpcId": {"Type": "AWS::EC2::VPC::Id"},
            "SubnetId": {"Type": "AWS::EC2::Subnet::Id"},
            "AvailabilityZone": {"Type": "AWS::EC2::AvailabilityZone::Name"},
        },
        "Resources": {
            "DataVolume": {
                "Type": "AWS::EC2::Volume",
                "DeletionPolicy": "Retain",
                "UpdateReplacePolicy": "Retain",
                "Properties": {
                    "AvailabilityZone": {"Ref": "AvailabilityZone"},
                    "Size": 80,
                    "VolumeType": "gp3",
                    "Iops": 3000,
                    "Throughput": 125,
                    "Encrypted": True,
                    "Tags": tags + [{"Key": "Name", "Value": "dockyard-cpu-data"}],
                },
            },
            "SecurityGroup": {
                "Type": "AWS::EC2::SecurityGroup",
                "Properties": {
                    "GroupDescription": "Dockyard CPU: no inbound ports; Session Manager access",
                    "VpcId": {"Ref": "VpcId"},
                    "SecurityGroupEgress": [{"IpProtocol": "-1", "CidrIp": "0.0.0.0/0"}],
                    "Tags": tags + [{"Key": "Name", "Value": "dockyard-cpu"}],
                },
            },
            "Instance": {
                "Type": "AWS::EC2::Instance",
                "Properties": {
                    "ImageId": {"Ref": "AmiId"},
                    "InstanceType": "r6a.large",
                    "AvailabilityZone": {"Ref": "AvailabilityZone"},
                    "IamInstanceProfile": "dockyard-host",
                    "MetadataOptions": {
                        "HttpTokens": "required",
                        "HttpPutResponseHopLimit": 1,
                        "HttpProtocolIpv6": "disabled",
                    },
                    "NetworkInterfaces": [{
                        "DeviceIndex": "0",
                        "AssociatePublicIpAddress": True,
                        "SubnetId": {"Ref": "SubnetId"},
                        "GroupSet": [{"Ref": "SecurityGroup"}],
                    }],
                    "BlockDeviceMappings": [{
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": 20,
                            "VolumeType": "gp3",
                            "Encrypted": True,
                            "DeleteOnTermination": True,
                        },
                    }],
                    "PropagateTagsToVolumeOnCreation": True,
                    "Tags": tags + [{"Key": "Name", "Value": "dockyard-cpu"}],
                    "UserData": {
                        "Fn::Base64": {
                            "Fn::Join": ["", [
                                "#!/usr/bin/env bash\nset -euo pipefail\nbash -s -- '",
                                {"Ref": "DataVolume"},
                                "' <<'DOCKYARD_BOOTSTRAP'\n",
                                bootstrap,
                                "\nDOCKYARD_BOOTSTRAP\n",
                            ]],
                        },
                    },
                },
            },
            "DataAttachment": {
                "Type": "AWS::EC2::VolumeAttachment",
                "Properties": {
                    "InstanceId": {"Ref": "Instance"},
                    "VolumeId": {"Ref": "DataVolume"},
                    "Device": "/dev/sdf",
                },
            },
        },
        "Outputs": {
            "InstanceId": {"Value": {"Ref": "Instance"}},
            "DataVolumeId": {"Value": {"Ref": "DataVolume"}},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(template(), indent=2) + "\n")
