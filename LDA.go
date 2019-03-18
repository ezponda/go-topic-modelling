package lda

type LDA struct {
	alpha float64
	beta  float64
}

func getNd(Ndx [][]int64) []int64 {
	D := len(Ndx)
	var rowSum int64
	Nd := make([]int64, D, D)
	for d, row := range Ndx {
		rowSum = 0
		for _, elem := range row {
			rowSum += elem
		}
		Nd[d] = rowSum
	}
	return Nd
}
func getNt(Nwt [][]int64) []int64 {
	T := len(Nwt[0])
	Nt := make([]int64, T, T)
	for _, row := range Nwt {
		for t, nwt := range row {
			Nt[t] += nwt
		}
	}
	return Nt
}

func get_wd(Ndw [][]int64) ([]int64, []int64) {

	var N int64
	for _, row := range Ndw {
		N += row[2]
	}

	w := make([]int64, N, N)
	d := make([]int64, N, N)
	var count int64
	for j := 0; j < len(Ndw); j++ {
		for i := count; i < count+Ndw[i][2]; i++ {
			w[i] = Ndw[j][1]
			d[i] = Ndw[j][0]
		}
		count += Ndw[j][2]
	}
	return w, d
}
