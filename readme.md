# How to run

Firstly we need to convert all of the source code into .mar file

just use 
```
python methods/torchserve2mar.py
```

Checkout dockerfile to see how to install the required libraries 

Or just use the docker version

```
sudo docker build -t streamo .
```

and then to run it add network host option

```
sudo docker run --network host --gpus all streamo 
```


## Important links

- http://127.0.0.1:9003/metrics -- metrics and basic info
- http://127.0.0.1:9001/predictions/all_det -- to predict basically send an image and return the model response


### example of predicting something 

```
binary_image = open("test.png", "rb").read()
r = requests.put(
            "http://127.0.0.1:9001/predictions/all_det", 
            binary_image, 
            timeout=timeout
        ).content

```
