terraform {
  backend "s3" {
    bucket = "terraform-state-stream-torch"
    key    = "cluster/terraform.tfstate"
    region = "us-east-2"
    }

}
