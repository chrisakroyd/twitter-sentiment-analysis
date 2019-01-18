import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';

import { interpolateHcl } from 'd3-interpolate';
import './word-heat.scss';

const highColour = '#ff6d77';
// const lowColour = '#fe9f60';
const lowColour = '#f2e5f1';

const interpolate = interpolateHcl(lowColour, highColour);

class WordHeat extends React.Component {
  generateWordComponents() {
    const { scores } = this.props;

    return this.props.tokens.map((word, i) =>
      (
        <div
          key={shortid.generate()}
          className="heat-word"
          style={({ backgroundColor: interpolate(scores[i] * 15) })}
        >
          {word}
        </div>));
  }


  render() {
    return (
      <div className="word-heat">
        <div className="heat-word-list">
          {this.generateWordComponents()}
        </div>
      </div>
    );
  }
}

WordHeat.propTypes = {
  tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  scores: PropTypes.arrayOf(PropTypes.number).isRequired,
};

export default WordHeat;
