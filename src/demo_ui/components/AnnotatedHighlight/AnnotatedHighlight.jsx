import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';

import './annotated-highlight.scss';

import annotations from '../../constants/annotations';

const annotatedWords = new Set(annotations);

class AnnotatedHighlight extends React.Component {
  generateWordComponents() {
    const splitWords = this.props.words.split(' ');
    return splitWords.map((word) => {
      if (annotatedWords.has(word)) {
        return (<div key={shortid.generate()} className="colour-word">{word}</div>);
      }
      return ` ${word} `;
    });
  }

  render() {
    return (
      <div className="annotated-highlight">
        <p>
          {this.generateWordComponents()}
        </p>
      </div>
    );
  }
}

AnnotatedHighlight.propTypes = {
  words: PropTypes.string.isRequired,
};

export default AnnotatedHighlight;
