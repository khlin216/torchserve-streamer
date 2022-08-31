terraform {
  backend "s3" {
    bucket = "terraform-streamtorch"
    key    = "helm/terraform.tfstate"
    region = "us-east-2"

  }
}

