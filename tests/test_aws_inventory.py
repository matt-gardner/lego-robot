import pytest

from so101.aws_inventory import require_no_gpu


def request_for(types, descriptions):
    def request(region, service, operation, *args):
        if operation == "describe-regions":
            return {"Regions": [{"RegionName": "us-west-2"}, {"RegionName": "us-east-1"}]}
        if operation == "describe-instances":
            return {"Reservations": [{"Instances": [{"InstanceType": kind} for kind in types]}]}
        if operation == "describe-instance-types":
            return {"InstanceTypes": descriptions}
        raise AssertionError(operation)
    return request


def test_cpu_only_and_empty_enabled_regions_pass():
    require_no_gpu(request_for(["t3.small"], [{"InstanceType": "t3.small"}]))
    require_no_gpu(request_for([], []))


@pytest.mark.parametrize("gpu", ["g6.2xlarge", "g5.2xlarge", "p5.48xlarge", "g4ad.xlarge"])
def test_all_gpu_types_deny_not_just_approved_nvidia_types(gpu):
    with pytest.raises(ValueError, match="Existing GPU"):
        require_no_gpu(request_for([gpu], [{"InstanceType": gpu, "GpuInfo": {"Gpus": []}}]))


@pytest.mark.parametrize("descriptions", [
    [], [{"InstanceType": "other.large"}],
    [{"InstanceType": "t3.small"}, {"InstanceType": "t3.small"}], [None],
])
def test_incomplete_or_malformed_classification_denies(descriptions):
    with pytest.raises(ValueError):
        require_no_gpu(request_for(["t3.small"], descriptions))


def test_inventory_is_scoped_to_approved_oregon_region():
    base = request_for([], [])
    calls = []
    def request(region, service, operation, *args):
        calls.append((region, operation))
        if region == "us-east-1":
            if operation == "describe-instances":
                return {"Reservations": [{"Instances": [{"InstanceType": "g5.2xlarge"}]}]}
            return {"InstanceTypes": [{"InstanceType": "g5.2xlarge", "GpuInfo": {}}]}
        return base(region, service, operation, *args)
    require_no_gpu(request)
    assert ("us-west-2", "describe-instances") in calls
    assert all(region == "us-west-2" for region, operation in calls)


def test_inventory_access_error_propagates():
    def denied(*args):
        raise PermissionError("inventory read denied")
    with pytest.raises(PermissionError, match="denied"):
        require_no_gpu(denied)
