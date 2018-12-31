/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;  // TODO: Set the number of particles
	
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	
	for (unsigned i = 0; i < particles.size(); i++) {
		Particle p = particles[i];
		double x_p, y_p, theta_p;

		// Check for division by zero
		if (fabs(yaw_rate) < 0.001) {
			x_p = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			y_p = p.y + velocity/yaw_rate + (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
		} else {
			x_p = p.x + velocity * delta_t * cos(p.theta);
			y_p = p.y + velocity * delta_t * sin(p.theta);
		}

		theta_p = p.theta + delta_t * yaw_rate;

		p.x = x_p;
		p.y = y_p;
		p.theta = theta_p;

		// Add noise to new x, y, and theta
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned i = 0; i < observations.size(); i++) {
		LandmarkObs l = observations[i];

		// By using the max double value we ensure that as long as there's 1 predicted LandmarkObs we cam
		// get a "nearest neighbor"
		double min_d = std::numeric_limits<double>::max();
		
		// id corresponding to nearest neighbor
		int nn_id = -1;

		// Find nearest neighbor
		for (unsigned k = 0; k < predicted.size(); k++) {
			LandmarkObs l_p = predicted[k];

			double d = dist(l.x, l.y, l_p.x, l_p.y);

			if (d < min_d) {
				min_d = d;
				nn_id = l_p.id;
			}
		}

		if (nn_id == -1) {
			throw std::invalid_argument("Each observed landmark must have a nearest neighbor");
		}

		observations[i].id = nn_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (unsigned i = 0; i < particles.size(); i++) {
		Particle p = particles[i];
		
		// Transform each observed landmark coordinate pair from car to map
		std::vector<LandmarkObs> t_obs;
		for (unsigned int j = 0; j < observations.size(); j++) {
			LandmarkObs o = observations[j];
			o.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);
			o.y = p.y + (sin(p.theta) * o.x) - (cos(p.theta) * o.y);
		}

		// Find the landmarks within sensor range to current particle
		std::vector<LandmarkObs> possibilities;
		for (unsigned int k = 0; k < map.landmark_list.size(); k++) {
			const float l_x = map.landmark_list[k].x_f;
			const float l_y = map.landmark_list[k].y_f;

			if (dist(p.x, p.y, l_x, l_y) <= sensor_range) {
				LandmarkObs l;
				l.x = l_x;
				l.y = l_y;
				l.id = map.landmark_list[k].id_i;
				possibilities.push_back(l);
			}
		}

		// Connect nearest neightbor
		dataAssociation(possibilities, t_obs);

		// Reset weight to 1.0 so we have leverage "*="
		particles[i].weight = 1.0;

		// Loop through transformed observations and find matching landmark in the map
		// We only need to search in possibilities, proof is left as exercise for the reader
		for (unsigned int k = 0; k < t_obs.size(); k++) {
			LandmarkObs l_o = t_obs[k];

			// Values for the possiblity with matching landmark id
			double p_x, p_y;

			for (unsigned int l = 0; l < possibilities.size(); l++) {
				if (possibilities[l].id == l_o.id) {
					p_x = possibilities[l].x;
					p_y = possibilities[l].y;
					break;
				}
			}

			particles[i].weight *= multivariate_gauss_prob_dens(l_o.x, l_o.y, p_x, p_y, std_landmark[0], std_landmark[1]);
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Get weights on each particle
	std::vector<double> p_weights;
	for (unsigned i = 0; i < particles.size(); i++) {
		p_weights.push_back(particles[i].weight);
	}

	// Get random index to start "resampling wheel"
	default_random_engine gen;
	uniform_int_distribution<int> index_dist(0, int(particles.size()) - 1);
	int index = index_dist(gen);

	double beta = 0.0;
	double max = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> weight_dist(0.0, max);

	// Resampled particles
	std::vector<Particle> rs_particles;

	for (unsigned i = 0; i < weights.size(); i++) {
		beta += weight_dist(gen) * 2.0;

		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % int(weights.size());
		}
		rs_particles.push_back(particles[index]);
	}

	particles = rs_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
