import React from 'react';
import PropTypes from 'prop-types';
import * as d3 from 'd3';
import { withFauxDOM } from 'react-faux-dom';

import './donut-chart.scss';

const width = 250;
const height = 200;
const animDuration = 1000;

class DonutChart extends React.Component {
  componentDidMount() {
    const faux = this.props.connectFauxDOM('div', 'chart');
    const dataset = this.preprocessData(this.props.statistics);
    const outerRadius = Math.min(width - 50, height - 50) / 2;
    const innerRadius = outerRadius - 15;
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const pie = d3.pie()
      .value(d => d.percent)
      .sort(null)
      .padAngle(0.03);

    const arc = d3.arc()
      .outerRadius(outerRadius)
      .innerRadius(innerRadius);

    const outerArc = d3.arc()
      .outerRadius(outerRadius * 1.3)
      .innerRadius(innerRadius * 1.2);

    const svg = d3.select(faux)
      .append('svg')
      .attr('width', 205)
      .attr('height', 200)
      .attr('class', 'shadow')
      .append('g')
      .attr('transform', `translate(${(width - 50) / 2}, ${height / 2})`);

    const path = svg.selectAll('path')
      .data(pie(dataset))
      .enter()
      .append('path')
      .attr('d', arc)
      .attr('fill', (d, i) => color(d.data.name));

    path.transition()
      .duration(animDuration)
      .attrTween('d', (d) => {
        const interpolate = d3.interpolate({ startAngle: 0, endAngle: 0 }, d);
        return t => arc(interpolate(t));
      });

    const text = svg.selectAll('text')
      .data(pie(dataset))
      .enter()
      .append('text')
      .transition()
      .duration(200)
      .attr('transform', (d) => {
        const coords = outerArc.centroid(d);
        coords[0] -= 10;
        return `translate(${coords})`;
      })
      .attr('dy', '.2em')
      .attr('text-anchor', 'middle')
      .text(d => `${d.data.percent}%`)
      .style('fill', '#000')
      .style('font-size', '10px');

    const legendRectSize = 15;
    const legendSpacing = 4;
    const legendHeight = legendRectSize + legendSpacing;

    const legend = svg.selectAll('.legend')
      .data(color.domain())
      .enter()
      .append('g')
      .attr('class', 'legend')
      .attr('transform', (d, i) => `translate(-40, ${((i * legendHeight) - 32)})`);

    legend.append('rect')
      .attr('width', legendRectSize)
      .attr('height', legendRectSize)
      .attr('rx', 10)
      .attr('ry', 10)
      .style('fill', color)
      .style('stroke', color)
      // .attr('transform', (d, i) => `translate(-0, ${(legendHeight)})`);
      .attr('transform', (d, i) => `translate(+8, +4)`);

    legend.append('text')
      .attr('x', 30)
      .attr('y', 15)
      .text(d => d)
      .style('fill', '#929DAF')
      .style('font-size', '11px');

    this.props.animateFauxDOM(animDuration);
  }

  preprocessData(statistics) {
    const total = statistics.reduce((accumulator, curr) =>
      ({ count: accumulator.count + curr.count })).count;
    return statistics.map(stat =>
      Object.assign({}, stat, { percent: Math.round((stat.count / total) * 100) }));
  }

  render() {
    return (
      <div className="donut-chart">
        {this.props.chart}
      </div>
    );
  }
}

DonutChart.propTypes = {
  statistics: PropTypes.arrayOf(PropTypes.shape({
    name: PropTypes.string.isRequired,
    count: PropTypes.number.isRequired,
  })).isRequired,
};

export default withFauxDOM(DonutChart);
