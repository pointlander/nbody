// Copyright 2026 The nbody Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
}

// LearnEmbedding learns the embeddings
func LearnEmbedding(iris []Fisher, size, width, iterations int) []Fisher {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	cp := make([]Fisher, length)
	copy(cp, iris)
	others.Add("x", size, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .1
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range iterations {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				_ = mhat
				// w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				if rng.Float64() > .01 {
					w.X[ii] -= .05 * g
				} else {
					w.X[ii] += .05 * g
				}
			}
		}
		//fmt.Println(l)
	}

	ii := set.ByName["i"]
	for i := range ii.S[1] {
		embedding := make([]float64, ii.S[0])
		copy(embedding, ii.X[i*ii.S[0]:(i+1)*ii.S[0]])
		cp[i].Embedding = embedding
	}

	return cp
}

func main() {
	type Point struct {
		X, Y   float64
		VX, VY float64
		FX, FY float64
	}
	points := []Point{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 1, Y: 1},
	}
	for range 33 {
		input := make([]Fisher, len(points))
		for i := range input {
			input[i].Measures = make([]float64, len(points))
		}
		for i := range points {
			for j := range points {
				x := points[j].X - points[i].X
				y := points[j].Y - points[i].Y
				r := math.Sqrt(x*x + y*y)
				if r > 0 {
					input[i].Measures[j] = 1 / (r * r)
				}
			}
		}

		output := LearnEmbedding(input, len(points), len(points), 256)
		/*for i := range input {
			fmt.Println(input[i].Measures)
		}
		for i := range input {
			fmt.Println(input[i].Embedding)
		}
		fmt.Println()
		for i := range output {
			fmt.Println(output[i].Measures)
		}
		for i := range output {
			fmt.Println(output[i].Embedding)
		}*/
		for i := range points {
			for j := range points {
				x := points[j].X - points[i].X
				y := points[j].Y - points[i].Y
				r := math.Sqrt(x*x + y*y)
				if r > 0 {
					points[j].FX += output[i].Embedding[j] * x / r
					points[j].FY += output[i].Embedding[j] * y / r
				}
			}
		}
		for i := range points {
			dt := 1.0
			ax := points[i].FX
			ay := points[i].FY
			points[i].VX += ax * dt
			points[i].VY += ay * dt
			points[i].X += points[i].VX * dt
			points[i].Y += points[i].VY * dt
		}
		for i := range points {
			fmt.Println(points[i].X, points[i].Y)
		}
		fmt.Println()
		input = output
	}
}
