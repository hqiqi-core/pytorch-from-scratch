print(" ###### Test 1 ######")

model_name = "DINOv2"
learning_rate= 1e-4
batch_size = 32
epochs = 50
use_gpu = True
print("Model Name:",model_name,"Type:",type(model_name))
print("learning_rate:",learning_rate,"Type:",type(learning_rate))
print("batch_size:",batch_size,"Type:",type(batch_size))
print("epochs:",epochs,"Type:",type(epochs))
print("use_gpu:",use_gpu,"Type:",type(use_gpu))

print(" ###### Test 1 ######")
models = ["Resnet","ViT","Dinov2"]
print(f"Available models:model[0]:{models[0]},model[1]:{models[1]},model[2]:{models[2]},length:{len(models)}")

if use_gpu:
    print("Training device: GPU")
else :
    print("Training device: CPU")

models.append("CLIP")
print(models)

config = {
    "lr":1e-4,
    "batch_size": 32,
    "epochs": 50
}

print(config["lr"])
