const express = require('express');
const cors = require('cors');

const tweetGenerator = require('./tweetGenerator.js');
const processedTweetGenerator = require('./processedTweetGenerator.js');
const responseGenerators = require('./responseGenerators.js');

const { responseGenerator, pagedResponseGenerator } = responseGenerators;

const app = express();
const router = express.Router();

const defaultLimit = 10;

// Get model status

router.get('/status', (req, res) => {
  res.send({
    neuralStatus: {
      load: 0,
      model: '',
      graphicsCard: '',
      memoryUsage: 0,
    },
  });
});

// Get a random sample of tweets from either live api or train data.

router.get('/tweets/train/sample', (req, res) => {
  const data = tweetGenerator(req.query.limit || defaultLimit);
  const generatedResponse = pagedResponseGenerator(req, data);
  res.send(generatedResponse);
});

router.get('/tweets/live/sample', (req, res) => {
  const data = tweetGenerator(req.query.limit || defaultLimit);
  const generatedResponse = pagedResponseGenerator(req, data);
  res.send(generatedResponse);
});

// Process tweets
router.post('/tweets/process', (req, res) => {
  const data = processedTweetGenerator();
  const generatedResponse = responseGenerator(req, data);
  res.send(generatedResponse);
});

// Get a list of datasets/embeddings

router.get('/datasets/', (req, res) => {
  const data = fakeNewsGenerator();
  const generatedResponse = responseGenerator(req, data);
  res.send(generatedResponse);
});

router.get('/embeddings/', (req, res) => {
  const data = fakeNewsGenerator(req.query.limit || defaultLimit);
  const generatedResponse = responseGenerator(req, data);
  res.send(generatedResponse);
});

// list of models and results per model.

router.get('/models/', (req, res) => {
  const data = fakeNewsGenerator();
  const generatedResponse = responseGenerator(req, data);
  res.send(generatedResponse);
});

router.get('/models/results', (req, res) => {
  const data = fakeNewsGenerator();
  const generatedResponse = responseGeneratorr(req, data);
  res.send(generatedResponse);
});

app.use(cors());
app.use('/api/v1/', router);

app.enable('etag');

app.listen(8080, () => {
  console.log('started search server.');
});
