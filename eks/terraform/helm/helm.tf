# resource "helm_release" "k8s-device-plugin" {
#   name  = "k8s-device-plugin"
#   repository = "https://nvidia.github.io/k8s-device-plugin"
#   chart = "nvidia-device-plugin"
#   version = "0.6.0"
#   namespace = "kube-system"
#   wait = true
# }