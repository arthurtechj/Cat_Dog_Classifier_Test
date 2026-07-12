import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate(model, loader):

    model.eval()

    preds=[]
    labels=[]


    with torch.no_grad():

        for x,y in loader:

            x=x.cuda()

            output=model(x)

            prediction=torch.argmax(
                output,
                dim=1
            )

            preds.extend(
                prediction.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )


    print(
        "Accuracy:",
        accuracy_score(labels,preds)
    )

    print(
        "Precision:",
        precision_score(labels,preds)
    )

    print(
        "Recall:",
        recall_score(labels,preds)
    )

    print(
        "F1:",
        f1_score(labels,preds)
    )


    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(labels,preds)
    )