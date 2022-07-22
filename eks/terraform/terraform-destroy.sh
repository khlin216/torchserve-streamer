function run_destroy () {
    
    terraform destroy -auto-approve
}
export KUBE_CONFIG_PATH=~/.kube/config # IF YOU ARE USING WINDOWS THIS SHOULD BE DIFFERENT

## DONT RUN THIS IN DEV ONLY IN PROD
cd helm && run_destroy
cd ../cluster && run_destroy

