import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';
import * as d3 from 'd3';

const DONUT_HEIGHT = 250;
const DONUT_WIDTH = 450;
const colourMap = {
  0: '#5fcf80',
  1: 'gray',
  2: '#FF6666',
};

const labelMap = {
  0: 'Positive',
  1: 'Neutral',
  2: 'Negative',
};


function midAngle(d) {
  return d.startAngle + (d.endAngle - d.startAngle) / 2;
}

class Donut extends React.Component {
  render() {
    const radius = Math.min(DONUT_WIDTH, DONUT_HEIGHT) / 2;
    const pie = d3.pie().sort(null).value(d => d);
    const arc = d3.arc().innerRadius(radius * 0.8).outerRadius(radius * 0.65);
    const outerArc = d3.arc()
      .outerRadius(radius * 0.9)
      .innerRadius(radius * 0.9);
    const test = pie(this.props.probs);
    return (
      <svg ref={this.element} height={DONUT_HEIGHT} width={DONUT_WIDTH} transform="translate(225, 125)">
        <g className="labels">
          {
            test.map((prob, i) => {
              const pos = outerArc.centroid(prob);
              pos[0] = radius * 0.95 * (midAngle(prob) < Math.PI ? 1 : -1);
              const anchor = (midAngle(prob)) < Math.PI ? 'start' : 'end';
              return (
                <text
                  key={shortid.generate()}
                  dy=".35em"
                  textAnchor={anchor}
                  stroke={colourMap[i]}
                  transform={`translate(${pos})`}
                >
                  {labelMap[i]} {(prob.data * 100).toFixed(1)}%
                </text>);
            })
          }
        </g>
        <g className="lines">
          {
            test.map((prob, i) => {
              const pos = outerArc.centroid(prob);
              pos[0] = radius * 0.95 * (midAngle(prob) < Math.PI ? 1 : -1);
              const test2 = [arc.centroid(prob), outerArc.centroid(prob), pos];
              return <polyline key={shortid.generate()} points={test2} stroke={colourMap[i]} />;
            })
          }
        </g>
        {
          test.map((prob, i) => (
            <path
              key={shortid.generate()}
              d={arc(prob)}
              fill={colourMap[i]}
            />))
        }
      </svg>
    );
  }
}

Donut.propTypes = {
  probs: PropTypes.arrayOf(PropTypes.number).isRequired,
};

export default Donut;
