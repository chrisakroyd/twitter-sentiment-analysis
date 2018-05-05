const queryString = require('query-string');

const maxPageSize = 50;
const defaultPageSize = 10;


function getValidQuery(req) {
  return {
    search: req.query.search,
    filter: req.query.filter,
    page_size: req.query.page_size && req.query.page_size < maxPageSize ?
      parseInt(req.query.page_size, 10) : defaultPageSize,
    start: req.query.start ? parseInt(req.query.start, 10) : 0,
  };
}

function generatePageValues(query) {
  const nextQuery = {
    start: query.start + query.page_size,
    page_size: query.page_size,
  };

  const prevQuery = {
    start: query.start - query.page_size,
    page_size: query.page_size,
  };

  return { nextQuery, prevQuery };
}

function getAPIRoute(req, query, apiVersion = 'v1') {
  let apiRoute = null;

  if (query.start >= 0) {
    apiRoute = `/api/${apiVersion}${req.path}?${queryString.stringify(query)}`;
  }

  return apiRoute;
}

function responseGenerator(req, responseData = {}) {
  return {
    self: req.originalUrl,
    data: responseData,
    error_code: null,
    error_message: null,
  };
}


function pagedResponseGenerator(req, responseData = []) {
  const currQuery = getValidQuery(req);
  const { nextQuery, prevQuery } = generatePageValues(currQuery);

  return {
    // The URL sent, the url for the next page and the previous page, null if no next or prev page.
    self: req.originalUrl,
    next: getAPIRoute(req, nextQuery),
    prev: getAPIRoute(req, prevQuery),
    // Page related attributes for the paged response.
    start: currQuery.start,
    page_size: currQuery.page_size,
    page: Math.round(currQuery.start / currQuery.page_size),
    // Data for the response
    data: responseData,
    // Any errors.
    error_code: null,
    error_message: null,
  };
}

module.exports = { responseGenerator, pagedResponseGenerator };
