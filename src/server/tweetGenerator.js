const faker = require('faker');

function tweetGenerator(limit) {
  const tweets = [];

  for (let i = 0; i < limit; i++) {
    const generatedNewsItem = {
      username: faker.internet.userName(),
      text: faker.lorem.words(),
    };

    tweets.push(generatedNewsItem);
  }

  return tweets;
}

module.exports = tweetGenerator;
