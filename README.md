# sentiment_analysis_by_bert
sentiment analysis by bert

## Docker usage

### Building the Docker Image

First, ensure you have Docker installed. Then, navigate to the directory containing the Dockerfile in your command line and run the following command to build the Docker image:

```
docker build --rm -t {botname} .
```

### Running the Docker Container

Once the build is complete, you can use the following command to run the Docker container:

```
docker run --rm -d {botname}
```

You can also mount a local directory into the container for file interaction within the container using the `-v` parameter:

```
docker run -v /path/to/local/data:/app/data {botname}
```

### Customizing Startup Commands

If needed, you can customize the commands to execute when the container starts. Uncomment one line in `CMD` or `ENTRYPOINT` in the Dockerfile and build a new image.

## Notes

- If additional Python dependencies are required, add them to the requirements.txt file and rebuild the Docker image.
- If modifications to the application's code are necessary, make changes locally and rebuild the Docker image.
- To debug or perform other operations inside the container, you can use the `docker exec` command.

```