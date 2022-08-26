# EKS CONTROL PLANE
resource "aws_iam_role" "eks" {
  name = "${var.tag}-eks-cluster-role"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY

}

resource "aws_iam_policy" "ecr_policy" {
  name = "ecr-access-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ecr:*",
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}


resource "aws_iam_role_policy_attachment" "ecr-eks-attach" {
  
  role      = aws_iam_role.ecr_role.name
  policy_arn = aws_iam_policy.ecr_policy.arn
}

resource "aws_iam_role_policy_attachment" "eks-AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks.name
}

resource "aws_iam_role_policy_attachment" "eks-AmazonEKSServicePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.eks.name
}


resource "aws_eks_cluster" "eks" {
  name     = var.tag
  role_arn = aws_iam_role.eks.arn

  vpc_config {
    subnet_ids = data.aws_subnet_ids.default.ids
  }

  # Ensure that IAM Role permissions are created before and deleted after EKS Cluster handling.
  # Otherwise, EKS will not be able to properly delete EKS managed EC2 infrastructure such as Security Groups.
  depends_on = [
    aws_iam_role_policy_attachment.eks-AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.eks-AmazonEKSServicePolicy,
    aws_iam_role_policy_attachment.ecr-eks-attach,
    aws_iam_role_policy_attachment.ssm_attach,
  ]
}

output "endpoint" {
  value = aws_eks_cluster.eks.endpoint
}

output "kubeconfig-certificate-authority-data" {
  value = aws_eks_cluster.eks.certificate_authority.0.data
}

# NODE-GROUP
resource "aws_iam_role" "nodegroup" {
  name = "${var.tag}-eks-node-group-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role" "ssm_role" {
  name = "ssm_role"

  assume_role_policy =jsonencode({
     Statement = [{
      "Effect": "Allow",
      "Principal": {"Service": "ssm.amazonaws.com"},
      "Action": "sts:AssumeRole"
     }]
      Version = "2012-10-17"
     })
}

resource "aws_iam_role_policy_attachment" "ssm_attach" {
  role       = aws_iam_role.ssm_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "nodegroup-AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.nodegroup.name
}

resource "aws_iam_role_policy_attachment" "nodegroup-AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.nodegroup.name
}

resource "aws_iam_role_policy_attachment" "nodegroup-AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.nodegroup.name
}
<<<<<<< HEAD

resource "aws_eks_node_group" "nodegroup_gpu" {
  cluster_name    = aws_eks_cluster.eks.name
  node_group_name = "${var.tag}-group-gpu"
  node_role_arn   = aws_iam_role.nodegroup.arn
  subnet_ids      = data.aws_subnet_ids.default.ids
  instance_types = ["g4dn.xlarge"]
  ami_type = "AL2_x86_64_GPU"
  capacity_type = "ON_DEMAND"
  scaling_config {
    desired_size = 3
    max_size     = 3
    min_size     = 0
  }
  disk_size = 40 

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
