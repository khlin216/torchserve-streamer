provider "aws" {
  region     = "eu-west-1"
}

# Will we store our state in S3, and lock with dynamodb
# terraform {
#   backend "s3" {
#     # Replace this with your bucket name!
#     bucket         = "terraform-up-and-running-state-gg"
#     key            = "covid/prod/terraform.tfstate"
#     region         = "eu-west-3"
#     # Replace this with your DynamoDB table name!
#     dynamodb_table = "terraform-up-and-running-locks"
#     encrypt        = true
#   }
# }

# VARIABLE

variable "tag" {
  default = "stream-torch"
}
variable "owner" {
  default = "jafar"
} 
variable "repo" {
  default = "https://github.com/JafarBadour/torchserve-streamer"
}



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

