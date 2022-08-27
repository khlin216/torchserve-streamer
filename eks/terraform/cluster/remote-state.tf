terraform {
  backend "s3" {
    bucket = "terraform-state-stream-torch"
    key    = "cluster-state.tfstate"
    region = "us-east-2"
    }

}
