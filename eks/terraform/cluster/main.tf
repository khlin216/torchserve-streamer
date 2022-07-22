


# Fist get the default VPC and subnet IDs
data "aws_vpc" "default" {
  default = true
}

data "aws_subnet_ids" "default" {
  vpc_id = "${data.aws_vpc.default.id}"
}

# OUTPUTS

# output "default_vpc_id" {
#   value = "${data.aws_vpc.default.id}"
# }

# output "default_subnet_ids" {
#   value = ["${data.aws_subnet_ids.default.ids}"]
# }

# output "ecr_hello_container_registry" {
#   value = "${aws_ecr_repository.ecr.repository_url}"
# }


# output "iam_build_role_arn" {
#   value = aws_iam_role.build_role.arn
# }

