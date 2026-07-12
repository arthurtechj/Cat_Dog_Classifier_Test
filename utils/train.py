import torch
import time
from tqdm import tqdm


def train(model, train_loader, val_loader, epochs=5, lr=0.001):

    device="cuda"

    model.to(device)


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn=torch.nn.CrossEntropyLoss()


    start=time.time()


    history={
        "train_loss":[],
        "val_accuracy":[]
    }


    for epoch in range(epochs):

        model.train()

        total_loss=0


        for x,y in tqdm(train_loader):

            x=x.to(device)
            y=y.to(device)


            optimizer.zero_grad()

            output=model(x)

            loss=loss_fn(output,y)

            loss.backward()

            optimizer.step()


            total_loss+=loss.item()



        history["train_loss"].append(
            total_loss/len(train_loader)
        )


        print(
            f"Epoch {epoch+1}",
            "loss:",
            history["train_loss"][-1]
        )


    training_time=time.time()-start


    return history, training_time