import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './demo.scss';

import WordHeat from './WordHeat/WordHeat';
import Donut from './Donut/Donut';

import Button from './common/Button';
import InputBar from './common/InputBar';
import LoadingSpinner from './common/LoadingSpinner';

import { predictionShape, classesShape } from '../prop-shapes';

const ClassificationContainer = ({ prediction, toggleToken, text }) => {
  const textLabel = prediction.label.toLowerCase();
  const classificationClass = classNames('label-header', {
    positive: textLabel === 'positive',
    negative: textLabel === 'negative',
    neutral: textLabel === 'neutral',
  });

  return (
    <div>
      <WordHeat
        onClick={toggleToken}
        tokens={prediction.tokens}
        enabled={prediction.enabled}
        scores={prediction.attentionWeights}
      />
      <div className="classification-container">
        <Donut probs={prediction.probs} classes={text.classes}/>
        <div className="label-container">
          <h3>We think this text is...</h3>
          <h2 className={classificationClass}>{prediction.label}</h2>
        </div>
      </div>
    </div>
  );
};


const Demo = ({ process, toggleToken, setText, predictions, text, loadExample }) => {
  const { error } = predictions;
  const hasText = text.text.length > 0;
  const validInput = hasText && error === null;
  let errorContent = '';
  let classificationContent = (
    <ClassificationContainer
      toggleToken={toggleToken}
      prediction={predictions}
      text={text}
    />);

  if (predictions.loading) {
    classificationContent = <LoadingSpinner />;
  } else if (!validInput && !predictions.loading) {
    let errorMessage;
    if (!hasText) {
      errorMessage = 'Please enter valid text.';
    } else {
      errorMessage = error.errorMessage;
    }

    errorContent = (
      <div className="error-container">
        <p>Error: {errorMessage}</p>
      </div>
    );
  }

  return (
    <div className="demo-body">
      <div className="header">
        <h1>Sentiment Analysis - Demo</h1>
        <h3>What is this thing?</h3>
        <p>
          This is a demo of a deep learning network that determines how positive, negative or neutral a segment of
          text is, otherwise known as its sentiment. It was trained on 60,000 labelled tweets that form the
          complete SemEval 2017 dataset. By clicking on individual tokens, you can examine how they influence
          the classification of the text.
        </p>
      </div>
      <div className="body">
        <h4>1. Enter Text</h4>
        <div className="enter-text-row">
          <InputBar
            onEnter={process}
            onRefresh={loadExample}
            value={text.text}
            placeholder="Enter text here"
            onKeyPress={setText}
          />
          <Button onClick={process} label="Predict" enabled={validInput}/>
        </div>
        {errorContent}
        <div className="text-block">
          <h4>2. Attention and Classification</h4>
          <p>
            Attention is a technique which focuses on the most pertinent information
            within the input and calculates a per-word relevance score. Below is a
            visualisation of the scores for the text input as well as its classification
            and our confidence in it.
            The strength of the colour reflects the strength of its impact.
          </p>
          {classificationContent}
        </div>
      </div>
    </div>
  );
};

Demo.propTypes = {
  // Functions
  process: PropTypes.func.isRequired,
  toggleToken: PropTypes.func.isRequired,
  setText: PropTypes.func.isRequired,
  loadExample: PropTypes.func.isRequired,
  // Data
  predictions: predictionShape.isRequired,
  text: PropTypes.shape({
    text: PropTypes.string.isRequired,
    classes: classesShape.isRequired,
  }).isRequired,
};

ClassificationContainer.propTypes = {
  toggleToken: PropTypes.func.isRequired,
  prediction: predictionShape.isRequired,
  text: PropTypes.shape({
    text: PropTypes.string.isRequired,
    classes: classesShape.isRequired,
  }).isRequired,
};

export default Demo;
