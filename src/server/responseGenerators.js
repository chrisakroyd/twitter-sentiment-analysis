const queryString = require('query-string');
const faker = require('faker');
const etag = require('etag');

const maxLimit = 30;
const defaultLimit = 10;
// Faker query time generation.
const max = 3;
const min = 0.2;


function getValidQuery(req) {
  return {
    searchTerms: req.query.q,
    limit: req.query.limit && req.query.limit < maxLimit ? req.query.limit : defaultLimit,
    offset: req.query.offset ? req.query.offset : 0,
  };
}

function responseGenerator(req, responseData = []) {
  return {
    self: req.originalUrl,
    next: `${req.hostname}${req.path}`,
    etag: etag(responseData.toString()),
    data: responseData,
    queryTime: parseFloat(((Math.random() * (max - min + 1)) + min).toFixed(2)),
    totalResults: responseData.length,
  };
}


function pagedResponseGenerator(req, responseData = []) {
  const currQuery = getValidQuery(req);
  const nextQuery = {
    limit: currQuery.limit,
    offset: currQuery.offset + currQuery.limit,
  };

  return {
    self: req.originalUrl,
    next: `${req.hostname}${req.path}?${queryString.stringify(nextQuery)}`,
    etag: etag(responseData.toString()),
    data: responseData,
    queryTime: parseFloat(((Math.random() * (max - min + 1)) + min).toFixed(2)),
    totalResults: faker.random.number({ min: 60000, max: 180000 }),
    limit: nextQuery.limit,
    offset: nextQuery.offset,
  };
}

module.exports = { responseGenerator, pagedResponseGenerator };
