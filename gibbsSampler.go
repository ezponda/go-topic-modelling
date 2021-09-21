package lda

import (
	"math/rand"
)

func gibbsLDA(z []int64, w []int64, d []int64, Nwt [][]int64, Ndt [][]int64, Nt []int64,
	alpha float64, beta float64) {
	N := len(z)
	W := len(Nwt)
	T := len(Nwt[0])
	var cumprob, th float64
	var wi, di, t int64
	probs := make([]float64, T, T)
	for i := 0; i < N; i++ {
		wi = w[i]
		di = d[i]
		t = z[i]
		Nt[t]--
		Nwt[wi][t]--
		Ndt[di][t]--
		for j := 0; j < T; j++ {
			probs[j] = (float64(Nwt[wi][j]) + beta) * (float64(Ndt[di][j]) + alpha) / (float64(Nt[j]) + beta*float64(W))
		}
		// Sample topic from di,wi
		th = rand.Float64()
		cumprob = probs[0]
		t = 0
		for th > cumprob {
			t++
			cumprob += probs[t]
		}
		// Update topic
		z[i] = t
		Nt[t]++
		Nwt[wi][t]++
		Ndt[di][t]++
	}
}

func iterateGibbs(z []int64, w []int64, d []int64, Nwt [][]int64, Ndt [][]int64,
	alpha float64, beta float64, gibbsIter int, optiLag int) (float64, float64) {
	var Nt []int64
	for it := 0; it < gibbsIter; it++ {
		Nt = getNt(Nwt)
		gibbsLDA(z, w, d, Nwt, Ndt, Nt, alpha, beta)
		Nt = getNt(Nwt)
		if it%optiLag == 0 {
			alpha = OptimizeParameter(alpha, Ndt, 25, 1e-5)
			beta = OptimizeParameter(beta, Nwt, 25, 1e-5)
		}
	}
	return alpha, beta
}
