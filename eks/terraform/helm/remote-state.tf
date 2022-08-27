terraform {
  backend "s3" {
    bucket = "terraform-state-stream-torch"
    key    = "helm-state.tfstate"
    region = "us-east-2"

  }
}

