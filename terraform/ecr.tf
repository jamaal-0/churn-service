resource "aws_ecr_repository" "api" {
  name = "churn-api"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "mlflow" {
  name = "churn-mlflow"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "initialise" {
  name = "churn-initialise"

  image_scanning_configuration {
    scan_on_push = true
  }
}