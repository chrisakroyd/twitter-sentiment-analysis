import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import shortid from 'shortid';

import { interpolateHcl } from 'd3-interpolate';
import './word-heat.scss';

const highColour = '#ff6d77';
const lowColour = '#f2e5f1';

const interpolate = interpolateHcl(lowColour, highColour);

class WordHeat extends React.Component {
  generateWordComponents() {
    const { scores, onClick, enabled } = this.props;
    const maxScore = Math.max(...scores);
    return this.props.tokens.map((word, i) => {
      const style = {};
      if (enabled[i]) {
        style.backgroundColor = interpolate(scores[i] / maxScore);
      }
      return (
        <div
          key={shortid.generate()}
          role="button"
          tabIndex={0}
          onKeyPress={(event) => { if (event.key === 'Enter') onClick(i); }}
          className={classNames('heat-word', { enabled: enabled[i] })}
          onClick={() => onClick(i)}
          style={style}
        >
          {word}
        </div>);
    });
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
  onClick: PropTypes.func.isRequired,
  tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
  enabled: PropTypes.arrayOf(PropTypes.bool).isRequired,
  scores: PropTypes.arrayOf(PropTypes.number).isRequired,
};

export default WordHeat;
