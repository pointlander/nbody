// Copyright 2026 The nbody Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
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
	rng := rand.New(rand.NewSource(1))
	type Point struct {
		X, Y, Z    float64
		VX, VY, VZ float64
		FX, FY, FZ float64
	}
	points := []Point{}
	for range 33 {
		points = append(points, Point{X: rng.Float64(), Y: rng.Float64(), Z: rng.Float64()})
	}
	images := &gif.GIF{}
	var palette = []color.Color{}
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
	for range 256 {
		input := make([]Fisher, len(points))
		for i := range input {
			input[i].Measures = make([]float64, len(points))
		}
		for i := range points {
			for j := range points {
				x := points[j].X - points[i].X
				y := points[j].Y - points[i].Y
				z := points[j].Z - points[i].Z
				r := math.Sqrt(x*x + y*y + z*z)
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
			points[i].FX = 0
			points[i].FY = 0
			points[i].FZ = 0
		}
		for i := range points {
			for j := range points {
				x := points[j].X - points[i].X
				y := points[j].Y - points[i].Y
				z := points[j].Z - points[i].Z
				r := math.Sqrt(x*x + y*y + z*z)
				if r > 0 {
					points[j].FX += output[i].Embedding[j] * x / r
					points[j].FY += output[i].Embedding[j] * y / r
					points[j].FZ += output[i].Embedding[j] * z / r
				}
			}
		}
		for i := range points {
			dt := 1.0
			ax := points[i].FX
			ay := points[i].FY
			az := points[i].FZ
			points[i].VX += ax * dt
			points[i].VY += ay * dt
			points[i].VZ += az * dt
			points[i].X += points[i].VX * dt
			points[i].Y += points[i].VY * dt
			points[i].Z += points[i].VZ * dt
		}
		out := make([]float64, 0, 3*len(points))
		in := make([]float64, 0, 3*len(points))
		for i := range points {
			in = append(in, points[i].X, points[i].Y, points[i].Z)
		}
		data := mat.NewDense(len(points), 3, in)
		var pc stat.PC
		ok := pc.PrincipalComponents(data, nil)
		if !ok {
			panic("failed to compute principal components")
		}

		var projection mat.Dense
		var vector mat.Dense
		pc.VectorsTo(&vector)
		projection.Mul(data, &vector)
		rows, cols := projection.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out = append(out, projection.At(i, j))
			}
		}

		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range points {
			x, y := out[i*cols], out[i*cols+1]
			fmt.Println(x, y)
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
		image := image.NewPaletted(image.Rect(0, 0, 512, 512), palette)
		for i := range points {
			xx, yy := out[i*cols], out[i*cols+1]
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			image.Set(int(x), int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
		}
		images.Image = append(images.Image, image)
		images.Delay = append(images.Delay, 10)
		fmt.Println()
		input = output
	}
	out, err := os.Create("verse.gif")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	err = gif.EncodeAll(out, images)
	if err != nil {
		panic(err)
	}
}
