export KUBE_CONFIG_PATH=~/.kube/config # IF YOU ARE USING WINDOWS THIS SHOULD BE DIFFERENT

## DONT RUN THIS IN DEV ONLY IN PROD
cd helm
terraform destroy -auto-approve

cd ../cluster
terraform destroy -auto-approve
