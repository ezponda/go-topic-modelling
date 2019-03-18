package lda

import "math"

func OptimizeParameter(gamma float64, Ndx [][]int64, maxIter int, tol float64) float64 {
	// Newton's method for argmax_gamma product_d ( B( Ndx_{d,:} + gamma) / B(gamma)  )
	D := len(Ndx)
	X := float64(len(Ndx[0]))
	var Df64 = float64(D)
	Nd := getNd(Ndx)
	gradD := make([]float64, D, D)
	hessD := make([]float64, D, D)
	var grad float64
	var hess float64
	for i := 0; i < maxIter; i++ {
		for d := 0; d < D; d++ {
			gradD[d] = digammaSum(Ndx[d], gamma) - X*digamma(float64(Nd[d])+X*gamma)
			hessD[d] = polygamma1Sum(Ndx[d], gamma) - X*X*polygamma1(float64(Nd[d])+X*gamma)
		}
		grad = SumFloat64(gradD) - (X * Df64 * (digamma(gamma) + digamma(X*gamma)))
		hess = SumFloat64(hessD) - (X * Df64 * (polygamma1(gamma) + X*polygamma1(X*gamma)))
		gamma = gamma - 0.05*grad/hess
		if math.Abs(grad) <= tol {
			return gamma
		}
	}
	return gamma
}

func SumFloat64(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v
	}
	return sum
}

func SumInt64(x []int64) int64 {
	var sum int64
	for _, v := range x {
		sum += v
	}
	return sum
}

func digamma(x float64) float64 {
	var r float64
	for x <= 5 {
		r -= 1 / x
		x++
	}
	var f = 1 / (x * x)
	var sum = f * (-1/12.0 + f*(1/120.0+f*(-1/252.0+f*(1/240.0+
		f*(-1/132.0+f*(691/32760.0+f*(-1/12.0+f*3617/8160.0)))))))

	return r + math.Log(x) - 0.5/x + sum

}

func digammaSum(sli []int64, k float64) float64 {
	var sum float64
	for _, x := range sli {
		sum += digamma(float64(x) + k)
	}
	return sum
}

func polygamma1(x float64) float64 {
	var r float64
	for x <= 5 {
		r += 1 / (x * x)
		x++
	}
	var f = 1 / (x * x)
	var sum = (1 + 0.5/x + f*(1/6+f*(-1/30+f*(1/42+f*(-1/30+
		f*(5/66+f*(-691/2730+f*(7/6+f*(-3617/510))))))))) / x
	return r + sum

}

func polygamma1Sum(sli []int64, k float64) float64 {
	var sum float64
	for _, x := range sli {
		sum += polygamma1(float64(x) + k)
	}
	return sum
}
