
resource "aws_eks_node_group" "nodegroup_gpu" {
  cluster_name    = aws_eks_cluster.eks.name
  node_group_name = "${var.tag}-group-gpu"
  node_role_arn   = aws_iam_role.nodegroup.arn
  subnet_ids      = data.aws_subnet_ids.default.ids
  instance_types = ["g4dn.xlarge"]
  ami_type = "AL2_x86_64_GPU"
  capacity_type = "ON_DEMAND"
  disk_size = 30
  scaling_config {
    min_size     = 1 
    max_size     = 3
    desired_size = 1 
  }


  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  # Otherwise, EKS will not be able to properly delete EC2 Instances and Elastic Network Interfaces.
  depends_on = [
    aws_iam_role_policy_attachment.nodegroup-AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.ssm_attach,
  ]
}


resource "aws_eks_node_group" "nodegroup_gpu_spot" {
  cluster_name    = aws_eks_cluster.eks.name
  node_group_name = "${var.tag}-group-gpu-spot"
  node_role_arn   = aws_iam_role.nodegroup.arn
  subnet_ids      = data.aws_subnet_ids.default.ids
  instance_types = ["g4dn.xlarge"]
  ami_type = "AL2_x86_64_GPU"
  capacity_type = "SPOT"
  disk_size = 30
  scaling_config {
    min_size     = 0
    max_size     = 10 
    desired_size = 4
  }

  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  # Otherwise, EKS will not be able to properly delete EC2 Instances and Elastic Network Interfaces.
  depends_on = [
    aws_iam_role_policy_attachment.nodegroup-AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.ssm_attach,
  ]
}


resource "aws_eks_node_group" "nodegroup_cpu" {
  cluster_name    = aws_eks_cluster.eks.name
  node_group_name = "${var.tag}-group-cpu"
  node_role_arn   = aws_iam_role.nodegroup.arn
  subnet_ids      = data.aws_subnet_ids.default.ids
  instance_types = ["t3.medium"]
  
  capacity_type = "SPOT"
  scaling_config {
    desired_size = 0
    max_size     = 20
    min_size     = 0
  }

  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  # Otherwise, EKS will not be able to properly delete EC2 Instances and Elastic Network Interfaces.
  depends_on = [
    aws_iam_role_policy_attachment.nodegroup-AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.ssm_attach,
  ]
}

resource "aws_eks_node_group" "nodegroup_cpu_master" {
  cluster_name    = aws_eks_cluster.eks.name
  node_group_name = "${var.tag}-group-cpu_master"
  node_role_arn   = aws_iam_role.nodegroup.arn
  subnet_ids      = data.aws_subnet_ids.default.ids
  instance_types = ["c5.4xlarge"]
  
  capacity_type = "ON_DEMAND"
  scaling_config {
    desired_size = 2  
    max_size     = 4
    min_size     = 0
  }

  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  # Otherwise, EKS will not be able to properly delete EC2 Instances and Elastic Network Interfaces.
  depends_on = [
    aws_iam_role_policy_attachment.nodegroup-AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.nodegroup-AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.ssm_attach,
  ]
}


