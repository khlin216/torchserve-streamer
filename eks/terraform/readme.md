# Structure

```

├── iam_policy.json
├── k8s
│   ├── deleteme.yaml
│   ├── deployment.yaml
│   ├── nvidia-smi.yaml
│   ├── ssm.yaml
│   └── svc_elb.yaml
├── readme.md
└── terraform
    ├── cluster
    │   ├── ecr.tf
    │   ├── eks.tf
    │   ├── main.tf
    │   └── vars.tf
    ├── helm
    │   ├── helm.tf
    │   └── vars.tf
    ├── terraform-destroy.sh
    └── terraform-run.sh



```

You need to add the iam_policy to the user that will execute **terraform-destroy.sh**. First you need to write their credentials in ~/.aws/credentials (this is different for windows). then after installing **aws-cli**, **kubectl** and **terraform**. Run

```
bash terraform-run.sh
```

This should set up the cluster. After the cluster is created (it takes 20 minutes approx) you can checkout how many nodes in the cluster by:
```
kubectl get nodes -o wide
```

to check if the GPU is being recognized by eks check:

```
kubectl describe nodes  |  tr -d '\000' | sed -n -e '/^Name/,/Roles/p' -e '/^Capacity/,/Allocatable/p' -e '/^Allocated resources/,/Events/p'  | grep -e Name  -e  nvidia.com  | perl -pe 's/\n//'  |  perl -pe 's/Name:/\n/g' | sed 's/nvidia.com\/gpu:\?//g'  | sed '1s/^/Node Available(GPUs)  Used(GPUs)/' | sed 's/$/ 0 0 0/'  | awk '{print $1, $2, $3}'  | column -t

```