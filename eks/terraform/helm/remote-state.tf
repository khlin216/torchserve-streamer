terraform {
  backend "s3" {
    bucket = "terraform-state-stream-torch"
    key    = "helm/terraform.tfstate"
    region = "us-east-2"

  }
}

