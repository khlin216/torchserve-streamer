terraform {
  backend "s3" {
    bucket = "terraform-streamtorch"
    key    = "cluster/terraform.tfstate"
    region = "us-east-2"
    }

}
