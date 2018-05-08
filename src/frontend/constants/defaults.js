import PropTypes from "prop-types";

const tweets = [
  {
    tweetId: 620013074272137216,
    username: 'UNKNOWN',
    text: 'Call for reservations for lunch or dinner tomorrow (yep Sunday!). Happy to accommodate guests in town for the MISS USA Pageant 346-5100',
  }, {
    tweetId: 620380830989336576,
    username: 'UNKNOWN',
    text: 'Tune in to the 2015 MISS USA Pageant July 12 8p ET/5p PT in Baton Rouge LIVE on ReelzChannel',
  }, {
    tweetId: 678053702306013185,
    username: 'UNKNOWN',
    text: '@AdamPlatt1999 one-sided support for Russia, which the Iran deal may\'ve began addressing. 2. Not doing so to a certain extent would lead to',
  }, {
    tweetId: 678420166221234176,
    username: 'UNKNOWN',
    text: '@RYOmoha @PrinceShaarawy ur fact: Milan hv a good squad and in the real world fact: Milan is 8th Placer hehe',
  }, {
    tweetId: 678437570506792960,
    username: 'UNKNOWN',
    text: '@CaptainLauren48 For your 21st, I got you a date with Chris Evans.  Sound good?  lol  :P',
  },
];

const status = {
  connected: true,
  graphicsCard: 'Geforce GTX 1080 TI',
  load: 5.4,
  memoryUsage: 4.2,
  maxMemoryUsage: 11.0,
  loading: false,
};

const datasets = [
  {
    name: 'SemEval',
    size: 49568,
    words: 40694,
    bestModel: 'LSTM_Attention',
    bestF1Score: 72.4,
    statistics: [
      { name: 'Positive', count: 19625 },
      { name: 'Neutral', count: 22211 },
      { name: 'Negative', count: 7732 },
    ],
  }, {
    name: 'Sent140',
    size: 1604006,
    words: 473000,
    bestModel: 'LSTM_Attention',
    bestF1Score: 86.2,
    statistics: [
      { name: 'Positive', count: 800000 },
      { name: 'Neutral', count: 4006 },
      { name: 'Negative', count: 800000 },
    ],
  }, {
    name: 'SemEval2017',
    size: 20552,
    words: 257381,
    bestModel: 'LSTM_Attention',
    bestF1Score: 71.2,
    statistics: [
      { name: 'Positive', count: 7032 },
      { name: 'Neutral', count: 10299 },
      { name: 'Negative', count: 3221 },
    ],
  },
];

const embeddings = [
  {
    name: 'Glove',
    corpus: 'Twitter',
    dimensionality: [25, 50, 100, 200],
    vocabSize: 1193514,
    tokens: '27B',
    plotSrc: 'images/tsne-twitter-200d.png',
  }, {
    name: 'Glove',
    corpus: 'Common Crawl',
    dimensionality: [50, 100, 200, 300],
    vocabSize: 1917494,
    tokens: '42B',
    plotSrc: 'images/tsne-common-300d.png',
  },
];

const models = [
  {
    name: 'LSTM_Attention',
    size: '9,104,803 Parameters',
    timePerEpoch: 20,
    learningRate: 0.001,
    clipNorm: 5.0,
    optimizer: 'RMSProp',
    topK: 0,
    bestAccuracy: 70.3,
    bestF1Score: 72.4,
    lastRun: '',
    classes: 3,
    architecture: {
      embeddingDim: 200,
      rnnUnits: 150,
      rnnLayers: 2,
      biDirectional: true,
      rnnParams: {
        l2: 0.0001,
        dropout: 0.3,
      },
      attention: true,
      concPool: false,
      classes: 3,
    },
  }, {
    name: 'LSTM_ConcPool',
    size: '9,115,903 Parameters',
    timePerEpoch: 32,
    learningRate: 0.001,
    clipNorm: 5.0,
    topK: 10,
    optimizer: 'RMSProp',
    bestAccuracy: 69.2,
    bestF1Score: 70.1,
    lastRun: '',
    classes: 3,
    architecture: {
      embeddingDim: 200,
      rnnUnits: 150,
      rnnLayers: 2,
      biDirectional: true,
      rnnParams: {
        l2: 0.0001,
        dropout: 0.3,
      },
      attention: false,
      concPool: true,
      classes: 3,
    },
  },
];

const results = [
  {
    name: 'LSTM_Attention',
    lastRun: '9:12:04 AM 09/05/2018',
    lastModified: '11:22:45 PM 08/05/2018',
    runs: 12,
    trainF1: 78.46,
    valF1: 80.91,
    trainAcc: 14.48,
    valAcc: 14.42,
    trainLoss: 0.45,
    valLoss: 0.45,
    trainConfMatrixSrc: 'images/train_confusion_lstm_attention.png',
    valConfMatrixSrc: 'images/val_confusion_lstm_attention.png',
    tokenizationScheme: 'TokenizerV2Annotate',
    trainF1Change: -1.21,
    trainAccChange: -0.23,
    trainLossChange: -0.05,
    valF1Change: 0.64,
    valAccChange: 0.28,
    valLossChange: 0.03,
  },
];

export {
  tweets,
  status,
  datasets,
  embeddings,
  models,
  results,
};
