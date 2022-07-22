provider "aws" {
  region     = "us-east-2"
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
