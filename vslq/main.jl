using LinearAlgebra
using SparseArrays
using DelimitedFiles

@everywhere using QuantumSimulations
@everywhere import Statistics
@everywhere import SharedArrays

@everywhere function pulse(t, params)
	tp = params[1]; tf = params[2]
	n = floor(t/(tp + tf))
	f = 0
	if (n*(tp + tf) <= t && t <= ((n + 1)*tp + n*tf))
		for i in 3:size(params,1)
			f += abs(params[i]*sin((i - 2)*(t%(tp + tf))*pi/tp))
		end
	end
	return f
end

@everywhere function block(t, params)
	tp = params[1]; tf = params[2]
	n = floor(t/(tp + tf))
	f = params[3]
	if (n*(tp + tf) <= t && t <= ((n + 1)*tp + n*tf))
		f = params[4]
	end
	return f
end

@everywhere function flat(t, params)
	return params[1]
end

function truncateData(data::Array{Float64,2}, tp, tf)
	truncLength::Int64 = floor(data[1, size(data,2)]/(tp + tf)) + 1
	truncData = zeros(2, truncLength)
	index = 1
	for i in 1:size(data,2)
		if data[1,i]%(tp + tf) < 1e-6
			truncData[1, index] = data[1, i]
			truncData[2, index] = data[2, i]
			index += 1
		end
	end
	return truncData
end

function GRAPE(initialState, H0, Ht, target, tvec, params, dc, acc, numCoeff, maxit)
	fHam = zeros(2, size(tvec,1))
	if size(params[1],1) < numCoeff
		for i in 1:numCoeff - 1
			append!(params[1], 0.0)
			append!(params[2], 0.0)
		end
	end
	
	for i in 1:size(tvec,1)
		fHam[1, i] = pulse(tvec[i], params[1])
		fHam[2, i] = pulse(tvec[i], params[2])
	end
	data = tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
	F1 = data[2, size(data,2)]
	F2 = data[6, size(data,2)]
	F3 = minimum(data[6, :])
	F = F1*F2*F3
	F4 = 1 - data[7, size(data,2)]; F *= F4
	dFx = SharedArrays.SharedArray{Float64}(numCoeff); dFy = SharedArrays.SharedArray{Float64}(numCoeff)
	println("Initial [F1, F2, F3, F4, F] = [$(F1), $(F2), $(F3), $(F4), $(F)]")
	check = (1 - F)/acc
	it = 0
	while it < maxit && check > 1
		if check > 100 eps = dc
		elseif check > 10 eps = 0.1*dc
		else eps = 0.01*dc
		end
		# @sync Threads.@threads for j in 3:size(params[1],1)
		@sync @distributed for j in 3:size(params[1],1)
		# for j in 3:size(params[1],1)
			xParams = copy(params); yParams = copy(params)
			xfHam = copy(fHam); yfHam = copy(fHam)
			xParams[1][j] += eps; yParams[2][j] += eps
			for i in 1:size(tvec,1)
				xfHam[1, i] = pulse(tvec[i], xParams[1])
				yfHam[2, i] = pulse(tvec[i], yParams[2])
			end
			locData = tpEvolveState(initialState, H0, target, Ht, xfHam, tvec)
			locF1 = locData[2, size(locData,2)]
			locF2 = locData[5, size(locData,2)]
			locF3 = minimum(locData[5, :])
			locF = locF1*locF2*locF3
			dFx[j - 2] = locF - F
			locData = tpEvolveState(initialState, H0, target, Ht, yfHam, tvec)
			locF1 = locData[2, size(locData,2)]
			locF2 = locData[5, size(locData,2)]
			locF3 = minimum(locData[5, :])
			locF = locF1*locF2*locF3
			dFy[j - 2] = locF - F
		end
		params[1][3:size(params[1],1)] += dFx; params[2][3:size(params[2],1)] += dFy
		for i in 1:size(tvec,1)
			fHam[1, i] = pulse(tvec[i], params[1])
			fHam[2, i] = pulse(tvec[i], params[2])
		end
		data = tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
		F1 = data[2, size(data,2)]
		F2 = data[5, size(data,2)]
		F3 = minimum(data[5, :])
		F = F1*F2*F3
		F4 = 1 - data[7, size(data,2)]; F *= F4
		it += 1
	end

	println("Final [F1, F2, F3, F4, F] = [$(F1), $(F2), $(F3), $(F4), $(F)] \ninteration: $(it)")

	file = open("output/singleQubit_GRAPE.dat", "w")
	writedlm(file, data)
	close(file)
	file = open("output/singleQubit_GRAPE.pls", "w")
	writedlm(file, params)
	close(file)
end

function vslq()
	W = 0.035*2*pi; delta = 0.350*2*pi; ws = W + 0.5*delta; GamS = 0.03
	k0::Array{Complex{Float64},1} = [1,0,0]; k1::Array{Complex{Float64},1} = [0,1,0]; k2::Array{Complex{Float64},1} = [0,0,1]
	r0 = k0*k0'; r1 = k1*k1'; r2 = k2*k2'
	ks0::Array{Complex{Float64},1} = [1, 0]; ks1::Array{Complex{Float64},1} = [0, 1]
	rs0 = ks0*ks0'; rs1 = ks1*ks1'
	a = [0 1 0; 0 0 sqrt(2); 0 0 0]; as = [0 1; 0 0]

	kL0 = normalize(k2 + k0)
	kL1 = normalize(k2 - k0)
	rL0 = kL0*kL0'
	rL1 = kL1*kL1'

	eye = [1 0 0; 0 1 0; 0 0 1]
	eyye = [1 0; 0 1]

	pl0 = sparse(tensor(eyye, r0, eye, eyye)); pr0 = sparse(tensor(eyye, eye, r0, eyye))
	pl1 = sparse(tensor(eyye, r1, eye, eyye)); pr1 = sparse(tensor(eyye, eye, r1, eyye))
	pl2 = sparse(tensor(eyye, r2, eye, eyye)); pr2 = sparse(tensor(eyye, eye, r2, eyye))
	al = sparse(tensor(eyye, a, eye, eyye)); ar = sparse(tensor(eyye, eye, a, eyye))
	asl = sparse(tensor(as, eye, eye, eyye)); asr = sparse(tensor(eyye, eye, eye, as))
	Xl = (al*al + al'al')/sqrt(2)
	Xr = (ar*ar + ar'ar')/sqrt(2)
	Zl = pl2 - pl0
	Zr = pr2 - pr0
	YL = im*Xl*Zl*Zr

	HP = -W*Xl*Xr + 0.5*delta*(pl1 + pr1)
	HS = ws*(asl'asl + asr'asr)
	HXR = ar*asr + ar'asr'
	HXL = al*asl + al'asl'
	HYR = im*(ar'asr' - ar*asr)
	HYL = im*(al'asl' - al*asl)
	HX = al*asl + ar*asr + al'asl' + ar'asr'
	HY = im*(al'asl' + ar'asr' - al*asl - ar*asr)
	# Hvec = [HP, HS, HX]

	psiL0 = tensor(ks0, kL0, kL0, ks0)
	psiL0c = tensor(ks0, kL0, kL0, ks1)
	kERr1 = tensor(ks0, kL0, k1, ks0)

	rhoL0 = sparse(tensor(rs0, rL0, rL0, rs0))
	rhoL0g = sparse(tensor(eyye, rL0, rL0, eyye))
	ERr1 = sparse(tensor(rs0, rL0, r1, rs0))
	pLX = 0.5*Xl*(I + Xl*Xr)*(I - pl1)*(I - pr1)
	pLY = 0.5*YL*(I + Xl*Xr)*(I - pl1)*(I - pr1)

	# println(HP)

	tp = 40; tf = 86
	GamP = 1e-3/20
	paramsHam = [
		[tp, tf, 0.06],
		[tp, tf, 0]
	]
	paramsCOp = [
		[tp, tf, GamS, GamP],
		[GamP],
		[GamP],
		[tp, tf, GamS, GamP]
	]

	# tvec = collect(0:0.05:10000)
	tvec = collect(0:0.05:tp)
	# fHam = [copy(tvec), copy(tvec)]
	# fCop = [copy(tvec), copy(tvec), copy(tvec), copy(tvec)]
	fHam = zeros(2, size(tvec,1))
	fCop = zeros(4, size(tvec,1))
	for i in 1:size(tvec,1)
		fHam[1, i] = pulse(tvec[i], paramsHam[1])
		fHam[2, i] = pulse(tvec[i], paramsHam[2])
		fCop[1, i] = block(tvec[i], paramsCOp[1])
		fCop[2, i] = flat(tvec[i], paramsCOp[2])
		fCop[3, i] = flat(tvec[i], paramsCOp[3])
		fCop[4, i] = block(tvec[i], paramsCOp[4])
	end
	
	initialStates = [kERr1, psiL0]
	targetStates = [psiL0c, psiL0]
	H = HP + HS
	cOps = [asl, al, ar, asr]
	Ht = [HX, HY]

	# @time data = meEvolveState(initialStates, H, cOps, targetStates, Ht, fHam, fCop, tvec)
	# @time data = tpEvolveState(initialStates, H, targetStates, Ht, fHam, tvec)
	println("Entering GRAPE with $(nprocs()) processors")
	@time GRAPE(initialStates, H, Ht, targetStates, tvec, paramsHam, 1e-5, 1e-4, 20, 300)
	
	# truncData = truncateData(data, tp, tf)
	# file = open("locout/truncData.dat", "w")
	# writedlm(file, truncData)

	# file = open("locout/pulse.dat", "w")
	# writedlm(file, data)

	# close(file)

	
	# truncLength = Int(floor(tvec[size(tvec,1)]/(tp + tf)) + 1)
	# truncData = zeros(1,truncLength)
	# index = 1
	# for i in 1:size(tvec,1)
	# 	if tvec[i]%(tp + tf) < 1e-6
	# 		truncData[index] = tvec[i]
	# 		index += 1
	# 	end
	# end
	# println(tvec[size(tvec,1)], '\n', truncLength, '\n', index, '\n', size(tvec,1), '\n', truncData[size(truncData,1) - 1])

	# println(H)
end

function singleQubit()
	d = 2*pi*0.35
	k0::Array{Complex{Float64},1} = [1,0,0]; k1::Array{Complex{Float64},1} = [0,1,0]; k2::Array{Complex{Float64},1} = [0,0,1]
	r0 = k0*k0'; r1 = k1*k1'; r2 = k2*k2'
	ks0::Array{Complex{Float64},1} = [1, 0]; ks1::Array{Complex{Float64},1} = [0, 1]
	rs0 = ks0*ks0'; rs1 = ks1*ks1'
	a = [0 1 0; 0 0 sqrt(2); 0 0 0]; as = [0 1; 0 0]

	eye = [1 0 0; 0 1 0; 0 0 1]
	eyye = [1 0; 0 1]

	pq2 = tensor(r2, eyye)
	aq = tensor(a, eyye)
	ar = tensor(eye, as)

	HP = -d*pq2
	HX = aq*ar + aq'ar'
	HY = im*(aq'ar' - aq*ar)

	psi00 = tensor(k0, ks0)
	psi10 = tensor(k1, ks0)
	psi11 = tensor(k1, ks1)
	psi21 = tensor(k2, ks1)

	initialStates = [psi00, psi10]
	targetStates = [psi11, psi10, psi21]
	cOps = [aq, ar]
	Ht = [HX, HY]

	tp = 20; tf = 40
	GamQ = 0.0001
	GamR = 0.01
	paramsHam = [
		[tp, tf, 0.02],
		[tp, tf, 0]
	]
	paramsCOp = [
		[tp, tf, GamR, GamQ],
		[GamQ],
		[GamQ],
		[tp, tf, GamR, GamQ]
	]
	tvec = collect(0:0.05:tp)

	println("Entering GRAPE with $(nprocs()) processors")
	@time GRAPE(initialStates, HP, Ht, targetStates, tvec, paramsHam, 1e-5, 1e-4, 20, 2e6)

end

singleQubit()
# vslq()