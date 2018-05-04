const faker = require('faker');

function procssedTweetGenerator(limit) {
  const tweets = [];

  for (let i = 0; i < limit; i++) {
    const generatedTweets = {
      username: faker.internet.userName(),
      text: faker.lorem.words(),
      tokenized: faker.lorem.words(),
      processed: faker.lorem.words(),
      label: faker.random.arrayElement(['positive', 'negative', 'neutral']),
    };

    tweets.push(generatedTweets);
  }

  return tweets;
}

module.exports = procssedTweetGenerator;
