export KUBE_CONFIG_PATH=~/.kube/config # IF YOU ARE USING WINDOWS THIS SHOULD BE DIFFERENT

cd cluster 

terraform init
terraform apply -auto-approve

## DONT RUN THIS IN DEV ONLY IN PROD
cd ../helm 
echo UPDATING EKS CONFIG
aws eks update-kubeconfig --name stream-torch --region us-east-2

terraform init
terraform apply -auto-approve

echo finished 
