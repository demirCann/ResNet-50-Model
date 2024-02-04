import time
import torch
import torch.optim as optim
from models.resnet import resnet50
from utils.dataloader import load_cifar10


def evaluate_model(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')


def main():
    batch_size = 4
    num_epochs = 10
    subset_size = 1000  # Number of images to use for training

    trainloader, testloader = load_cifar10(batch_size, subset_size=subset_size)

    net = resnet50()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()  # Start the timer

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 200 == 199:  # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed')

    print('Finished Training')

    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    print(f'Total training time: {total_time / 60:.2f} minutes')

    # Save the trained model
    torch.save(net.state_dict(), 'resnet_cifar10.pth')
    print('Model saved as resnet_cifar10.pth')

    # Evaluate the model
    evaluate_model(net, testloader)


if __name__ == "__main__":
    main()
