import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';

import './tweets.scss';
import tweetShape from '../../../../prop-shapes/tweetShape';

class Tweets extends React.Component {
  onClick(text) {
    this.props.setText(text);
  }

  createTweetList() {
    return this.props.tweets.map(tweet =>
      (
        <li
          key={shortid.generate()}
          className="tweet"
          onClick={() => this.onClick(tweet.text)}
        >
          <div className="icon-container">
            <i className="material-icons">account_circle</i>
          </div>
          <div className="content-container">
            <div className="content-meta">
              {`${tweet.username} - ${tweet.tweetId}`}
            </div>
            <div className="content-text">
              {tweet.text}
            </div>
          </div>
        </li>));
  }

  render() {
    return (
      <div className="tweets-container">
        <ul className="tweet-list">
          {this.createTweetList()}
        </ul>
      </div>
    );
  }
}

Tweets.propTypes = {
  tweets: PropTypes.arrayOf(tweetShape).isRequired,
  setText: PropTypes.func.isRequired,
};

export default Tweets;
