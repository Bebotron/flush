using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Match
using BlackBoxOptim

@everywhere using QuantumSimulations
@everywhere import Statistics
@everywhere import SharedArrays

@everywhere function pulse(t, params)
	tf = params[1]; tp = params[2]
	n = floor(t/(tp + tf))
	f = 0
	if (n*(tp + tf) <= t && t <= ((n + 1)*tp + n*tf))
		for i in 3:size(params,1)
			f += params[i]*sin((i - 2)*(t%(tp + tf))*pi/tp)
		end
	end
	return f
end

@everywhere function block(t, params)
	tf = params[1]; tp = params[2]
	n = floor(t/(tp + tf))
	f = params[3]
	if (n*(tp + tf) <= t && t <= ((n + 1)*tp + n*tf))
		f = params[4]
	end
	return f
end

function printRuntime(runtime)
    d = Int(floor(runtime/(24*3600)))
    h = Int(floor(runtime/3600))%24
    m = Int(floor(runtime/60))%60
    s = runtime%60;
    println("TIME TO RUN:\n $(d)d $(h)h $(m)m $(s)s")
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

@everywhere function getLifetime(data, index, correction, cutoffi, cutofff)
	x1 = data[1, Int(floor(cutoffi*size(data,2)))]
	x2 = data[1, Int(floor(cutofff*size(data,2)))]
	y1 = data[index, Int(floor(cutoffi*size(data,2)))]
	y2 = data[index, Int(floor(cutofff*size(data,2)))]
	m = (y2 - y1)/(x2 - x1)
	y0 = y2 - x2*m
	T1 = (correction - y0)/m
	return T1
end

function pulseDiffEvo(system, initialState, H0, Ht, target, tvec, params, fidsLocations, numCoeff, searchSpace, maxit)
	paramsInit = copy(params)
	if size(params,2) < numCoeff
		for i in 1:numCoeff - 1
			params = hcat(params, [0; 0])
		end
	end
	fHam = zeros(2, size(tvec,1))
	for i in 1:size(tvec,1)
		fHam[1, i] = pulse(tvec[i], params[1,:])
		fHam[2, i] = pulse(tvec[i], params[2,:])
	end
	data = tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
	outString1 = "Initial ["
	fids = zeros(length(fidsLocations))
	for i in 1:length(fids)
		fids[i] = data[fidsLocations[i], size(data,2)]
		outString1 *= "F$(i), "
	end

	F = prod(fids)

	outString1 *= "F] = ["
	for fid in fids
		outString1 *= "$(fid), "
	end
	outString1 *= "$(F)]"
	println(outString1)

	pulseCoeff = vcat(params[1,3:end], params[2,3:end])

	function getFidelity(x)
		locfHam = zeros(2, size(tvec,1))
		for i in 1:size(tvec,1)
			locfHam[1, i] = pulse(tvec[i], vcat(params[1,1:2], x[1:20]))
			locfHam[2, i] = pulse(tvec[i], vcat(params[2,1:2], x[21:40]))
		end

		data = tpEvolveState(initialState, H0, target, Ht, locfHam, tvec)
		fids = zeros(length(fidsLocations))
		for i in 1:length(fids)
			fids[i] = data[fidsLocations[i], size(data,2)]
			# outString1 *= "F$(i), "
		end

		F = prod(fids)
		return (1 - F)
	end

	# results = bboptimize(getFidelity; SearchRange = [
	# 	(pulseCoeff[1] - rangeBig, pulseCoeff[1] + rangeBig), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall),
	# 	(pulseCoeff[21] - rangeSmall, pulseCoeff[21] + rangeSmall), (-rangeBig, rangeBig), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall)
	# ], MaxSteps=maxit)
	results = bboptimize(getFidelity; SearchRange = searchSpace, MaxSteps = maxit)

	outString2 = "Final ["
	for i in 1:length(fids)
		fids[i] = data[fidsLocations[i], size(data,2)]
		outString2 *= "F$(i), "
	end
	F = prod(fids)
	
	outString2 *= "F] = ["
	for fid in fids
		outString2 *= "$(fid), "
	end
	outString2 *= "$(F)]\n"
	println(outString2)

	file1 = open("output/$(system)-PULSEDE.dat", "w")
	writedlm(file1, data)
	close(file1)
	file2 = open("output/$(system)-PULSEDE.pls", "w")
	writedlm(file2, best_candidate(results))
	close(file2)
	file3 = open("output/$(system)-PULSEDE.log", "w")
	write(file3, outString1, "\n", outString2)
	close(file3)
end

function fixedDiffEvo(system, initialState, H0, Ht, target, tvec, params, index, correction, collapseOps, collapsePrimary, searchSpace, maxit)
	H = H0
	numFixed = length(collapsePrimary)
	fHam = copy(params[1])
	fCollapse = zeros(length(collapseOps))
	fCollapse[1:numFixed] = collapsePrimary; fCollapse[numFixed + 1:end] = params[2]
	
	data = meEvolveState(initialState, target, H0, collapseOps, fCollapse, Ht, fHam, tvec, false)

	cutoffi = 0.95; cutofff = 1
	if system == "singleQubit" T1 = data[index, end]
	else T1 = 1e-3getLifetime(data, index, correction, cutoffi, cutofff) end

	string1 = "Initial T1 =  $(T1), | Params :: $(params)\n"
	println(string1)

	function optimizeLifetime(x)
		locfHam = copy(x[1:length(params[1])])
		locfCollapse = copy(fCollapse)
		locfCollapse[numFixed + 1:end] = copy(x[length(params[1]) + 1:end])
		locData = meEvolveState(initialState, target, H0, collapseOps, locfCollapse, Ht, locfHam, tvec, false)
		if system == "singleQubit" locT1 = locData[index, end]
		else locT1 = 1e-3getLifetime(locData, index, correction, cutoffi, cutofff) end
		return locT1
	end

	results = bboptimize(optimizeLifetime; SearchRange = searchSpace, MaxSteps = maxit)

	string2 = "Final T1 = $(best_fitness(results)) | Params :: $(best_candidate(results))\n"
	println(string2)

	file1 = open("output/$(system)-FIXEDDE.dat", "w")
	writedlm(file1, data)
	close(file1)
	file2 = open("output/$(system)-FIXEDDE.log", "w")
	write(file2, string1*string2)
end

function pulseGRAPE(system, initialState, H0, Ht, target, tvec, params, fidsLocations, dc, acc, numCoeff, maxit)
	fHam = zeros(2, size(tvec,1))
	if size(params,2) < numCoeff
		for i in 1:numCoeff - 1
			params = hcat(params, [0; 0])
		end
	end
	
	for i in 1:size(tvec,1)
		fHam[1, i] = pulse(tvec[i], params[1,:])
		fHam[2, i] = pulse(tvec[i], params[2,:])
	end

	data = tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
	outString1 = "Initial ["
	fids = zeros(length(fidsLocations))
	for i in 1:length(fids)
		fids[i] = data[fidsLocations[i], size(data,2)]
		outString1 *= "F$(i), "
	end
	# fids[size(fids,1)] = minimum(data[fidsLocations[size(fids,1)], :])
	F = prod(fids)

	outString1 *= "F] = ["
	for fid in fids
		outString1 *= "$(fid), "
	end
	outString1 *= "$(F)]"
	println(outString1)
	dFx = SharedArrays.SharedArray{Float64}(numCoeff); dFy = SharedArrays.SharedArray{Float64}(numCoeff)
	check = (1 - F)/acc
	it = 0
	while it < maxit && check > 1
		if check > 100 eps = dc
		elseif check > 10 eps = 0.1*dc
		else eps = 0.01*dc
		end
		if it == Int(floor(0.5maxit)) eps *= 0.5 end
		if it == Int(floor(0.75maxit)) eps *= 0.25 end
		if it == Int(floor(0.9maxit)) eps *= 0.1 end
		@sync @distributed for j in 3:size(params,2)
			xParams = copy(params); yParams = copy(params)
			xfHam = copy(fHam); yfHam = copy(fHam)
			xParams[1, j] += eps; yParams[2, j] += eps
			for i in 1:size(tvec,1)
				xfHam[1, i] = pulse(tvec[i], xParams[1,:])
				yfHam[2, i] = pulse(tvec[i], yParams[2,:])
			end

			locData = tpEvolveState(initialState, H0, target, Ht, xfHam, tvec)
			locFids = copy(fids)
			for i in 1:size(locFids,1)
				locFids[i] = locData[fidsLocations[i], size(locData,2)]
			end
			# locFids[size(fids,1)] = minimum(locData[fidsLocations[size(fids,1)], :])
			locF = prod(locFids)

			dFx[j - 2] = locF - F

			locData = tpEvolveState(initialState, H0, target, Ht, yfHam, tvec)
			locFids = copy(fids)
			for i in 1:size(locFids,1)
				locFids[i] = locData[fidsLocations[i], size(locData,2)]
			end
			# locFids[size(fids,1)] = minimum(locData[fidsLocations[size(fids,1)], :])
			locF = prod(locFids)

			dFy[j - 2] = locF - F
		end
		params[1, 3:size(params,2)] += dFx; params[2, 3:size(params,2)] += dFy
		for i in 1:size(tvec,1)
			fHam[1, i] = pulse(tvec[i], params[1,:])
			fHam[2, i] = pulse(tvec[i], params[2,:])
		end

		data = tpEvolveState(initialState, H0, target, Ht, fHam, tvec)
		for i in 1:size(fids,1)
			fids[i] = data[fidsLocations[i], size(data,2)]
		end
		# fids[size(fids,1)] = minimum(data[fidsLocations[size(fids,1)], :])
		F = prod(fids)
		check = (1 - F)/acc
		it += 1
	end

	outString2 = "Final ["
	for i in 1:length(fids)
		fids[i] = data[fidsLocations[i], size(data,2)]
		outString2 *= "F$(i), "
	end
	# fids[size(fids,1)] = minimum(data[fidsLocations[size(fids,1)], :])
	F = prod(fids)
	
	outString2 *= "F] = ["
	for fid in fids
		outString2 *= "$(fid), "
	end
	outString2 *= "$(F)]"
	println(outString2)
	println("Number of iterations: $(it)")

	file1 = open("output/$(system)-PULSEGRAPE.dat", "w")
	writedlm(file1, data)
	close(file1)
	file2 = open("output/$(system)-PULSEGRAPE.pls", "w")
	writedlm(file2, params)
	close(file2)
	file3 = open("output/$(system)-PULSEGRAPE.log", "w")
	write(file3, outString1, "\n", outString2, "\nNumber of iterations: $(it)\n")
	close(file3)
end

function fixedGRAPE(system, initialState, H0, Ht, target, tvec, params, index, dc, correction, collapse, collapsePrimary, maxit)
	H = H0
	numFixed = length(collapsePrimary)
	fHam = copy(params[1])
	fCollapse = zeros(length(collapse))
	fCollapse[1:numFixed] = collapsePrimary; fCollapse[numFixed + 1:end] = params[2]
	
	data = meEvolveState(initialState, target, H0, collapse, fCollapse, Ht, fHam, tvec, false)

	cutoffi = 0.95; cutofff = 1
	if system == "singleQubit" T1 = data[index,end]
	else T1 = 1e-3getLifetime(data, index, correction, cutoffi, cutofff) end
	# T1 = data[index,end]

	string1 = "Initial T1 =  $(T1), | Params :: $(params)\n"
	println(string1)
	
	dT1 = SharedArrays.SharedArray{Float64}(length(params[1]) + length(params[2]))
	it = 0
	eps = dc
	while it < maxit
		if it == Int(floor(0.5maxit)) eps *= 0.5 end
		if it == Int(floor(0.75maxit)) eps *= 0.25 end
		if it == Int(floor(0.9maxit)) eps *= 0.1 end
		@sync @distributed for j in 1:(length(params[1]) + length(params[2]))
			locParams = deepcopy(params)
			locfCollapse = copy(fCollapse)
			if j <= length(params[1])
				locParams[1][j] += eps
			else
				locParams[2][j - length(params[1])] += eps
			end
			locfHam = copy(locParams[1])
			locfCollapse[numFixed + 1:end] = locParams[2]
			
			locData = meEvolveState(initialState, target, H, collapse, locfCollapse, Ht, locfHam, tvec, false)

			if system == "singleQubit" locT1 = locData[index,end]
			else locT1 = 1e-3getLifetime(locData, index, correction, cutoffi, cutofff) end
			# locT1 = locData[index,end]
			dT1[j] = locT1 - T1
		end
		params[1] += dT1[1:length(params[1])]; params[2] += dT1[length(params[1]) + 1:end]
		fHam = params[1]; fCollapse[numFixed + 1:end] = params[2]

		data = meEvolveState(initialState, target, H, collapse, fCollapse, Ht, fHam, tvec, false)
		if system == "singleQubit" T1 = data[index,end]
		else T1 = 1e-3getLifetime(data, index, correction, cutoffi, cutofff) end
		# T1 = data[index,end]
		it += 1
	end

	string2 = "ITERATION: $(it)\nFinal T1 = $(T1) | Params :: $(params)\n"
	println(string2)

	file1 = open("output/$(system)-FIXEDGRAPE.dat", "w")
	writedlm(file1, data)
	close(file1)
	file2 = open("output/$(system)-FIXEDGRAPE.log", "w")
	write(file2, string1*string2)
end

function vslq(action, timedep, T1index)
	W = 0.035*2pi; delta = 0.350*2pi
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
	pls1 = sparse(tensor(rs1, eye, eye, eyye)); prs1 = sparse(tensor(eyye, eye, eye, rs1))
	al = sparse(tensor(eyye, a, eye, eyye)); ar = sparse(tensor(eyye, eye, a, eyye))
	asl = sparse(tensor(as, eye, eye, eyye)); asr = sparse(tensor(eyye, eye, eye, as))
	Xl = (al*al + al'al')/sqrt(2)
	Xr = (ar*ar + ar'ar')/sqrt(2)
	Zl = pl2 - pl0
	Zr = pr2 - pr0
	YL = im*Xl*Zl*Zr

	HP = -W*Xl*Xr + 0.5*delta*(pl1 + pr1)
	HW = -W*Xl*Xr
	HP1 = 0.5(pl1 + pr1)
	HS = asl'asl + asr'asr
	HXR = ar*asr + ar'asr'
	HXL = al*asl + al'asl'
	HYR = im*(ar'asr' - ar*asr)
	HYL = im*(al'asl' - al*asl)
	HX = al*asl + ar*asr + al'asl' + ar'asr'
	HY = im*(al'asl' + ar'asr' - al*asl - ar*asr)

	psiL0 = tensor(ks0, kL0, kL0, ks0)
	psiL0c = tensor(ks0, kL0, kL0, ks1)
	psiERr1 = tensor(ks0, kL0, k1, ks0)
	psiINDERr = tensor(ks0, kL0, k1, ks1)
	rhoY = 0.25*(I + YL)*(I + Xl*Xr)*(I - pl1)*(I - pr1)*(I - pls1)*(I - prs1)

	rhoL0 = sparse(tensor(rs0, rL0, rL0, rs0))
	rhoL0g = sparse(tensor(eyye, rL0, rL0, eyye))
	rhoL1 = sparse(tensor(rs0, rL1, rL1, rs0))
	rhoL1g = sparse(tensor(eyye, rL1, rL1, eyye))
	ERr1 = sparse(tensor(rs0, rL0, r1, rs0))
	pLX = 0.5*Xl*(I + Xl*Xr)*(I - pl1)*(I - pr1)
	pLY = 0.5*YL*(I + Xl*Xr)*(I - pl1)*(I - pr1)

	# (rhoL0g - rhoL1g) == pLX   #<-- This is true, even though printing this statement will give false.

	tp = 40; tf = 0
	# GamP = 1e-3/20
	GamP = 1e-3./(10*[1,2,3,4,5,6,7,8,9,10])
	GamP_60 = 1e-3./(5*[1,2,3,4,5,6,7,8,9,10,11,12])
	# GamR_0 = 0.03
	GamR_0 = 0.0092
	Ohm_0 = 0.0055
	Ws_0 = W + 0.5*delta

	Ws_NEW = [1.31824, 1.31852, 1.31867, 1.31878, 1.31886, 1.31893, 1.31898, 1.31902, 1.31905, 1.31908]
	Ohm_NEW = [0.0261059, 0.0241047, 0.0229972, 0.0221587, 0.0214653, 0.0208711, 0.0203517, 0.0198914, 0.019479, 0.0191062]
	GamR_NEW = [0.036792, 0.0329765, 0.0315307, 0.0307923, 0.0303562, 0.0300751, 0.0298828, 0.0297457, 0.0296447, 0.0295685]

	Ws_OLD = 1e-3*[1318.51, 1318.88, 1319.00, 1319.09, 1319.14, 1319.19]
	Ohm_OLD = 1e-3*[21.80, 16.33, 13.30, 11.45, 10.19, 9.26]
	GamR_OLD = 1e-3*[20.08, 15.89, 14.37, 13.38, 12.59, 11.96]

	
	Ws_X = 1e-3*[1317.76,1318.51,1318.78,1318.88,1318.95,1319.00,1319.05,1319.09,1319.12,1319.14,1319.17,1319.19]
	Ohm_X = 1e-3*[25.52, 21.80, 18.74, 16.33, 14.62, 13.30, 12.27, 11.45, 10.76, 10.19, 9.69, 9.26]
	GamR_X = 1e-3*[30.20, 20.08, 17.07, 15.89, 15.01, 14.37, 13.84, 13.38, 12.97, 12.59, 12.26, 11.96]

	Ws_Y = 1e-3*[1317.93, 1318.44, 1318.89, 1319.00, 1319.12, 1319.19, 1319.23, 1319.26, 1319.28, 1319.30, 1319.32, 1319.33]
	Ohm_Y = 1e-3*[18.52, 13.51, 11.42, 10.01, 9.01, 8.28, 7.69, 7.21, 6.81, 6.47, 6.16, 5.90]
	GamR_Y = 1e-3*[24.66, 18.40, 15.32, 13.26, 12.09, 11.09, 10.30, 9.67, 9.15, 8.73, 8.38, 8.04]

	trVec = [75, 88, 102, 118, 118, 118, 118, 159, 159, 159, 159, 159]
	trVec40DE = [166, 167, 210, 210, 216, 224, 239, 239, 239, 239, 239, 287]

	
	# [166.0, 54.6759]
	# [167.0, 190.442]
	# [210.0, 397.993]
	# [210.0, 670.46]
	# [216.0, 999.855]
	# [224.0, 1380.06]
	# [239.0, 1807.8]
	# [239.0, 2281.1]
	# [239.0, 2793.31]
	# [239.0, 3340.93]
	# [239.0, 3920.87]
	# [287.0, 4543.16]
	

	trVecTEST = [ 238, 278, 297, 297, 297, 278,  278,  278,  278,  278,   99,   99]
	trVecTEST2 = [155, 183, 196, 196, 196, 210, 210, 210, 210, 210, 210, 210]

	H = HP + Ws_0*HS
	cops = [al, ar, asl, asr]
	Ht = [HX, HY]
	
	function regularEvolution()
		initialMats = [rhoL0]
		targetMats = [pLX]
		tp = 40

		# filename = "sol/X-tp=60-W=0.035-LONG-PR.dat"
		datafile = "sol/X-tp=$(tp)-W=$(W/(2pi))-fixed.dat"
		lifetimesfile = "sol/X-tp=$(tp)-W=$(W/(2pi))-fixed-lifetimes.dat"
		# filename = "output/X-FIXED-EVOLUTION.dat"
		# filename = "output/vslq-TR=$(tf)-FLUSHED-EVOLUTION-LONG.dat"
		# filename = "output/X-TR=TRVEC-FLUSHED-EVOLUTION-LONG.dat"
		# pulseParams = readdlm("output/vslq-tp=$(tp)-W=$(W/(2pi))-PULSEGRAPE.pls")
		pulseRaw = readdlm("output/vslq-tp=$(tp)-W=$(W/(2pi))-PULSEDE.pls")
		pulseX = hcat([tp 0], pulseRaw[1:20]')
		pulseY = hcat([tp 0], pulseRaw[21:end]')
		pulseParams = vcat(pulseX, pulseY)

		tvec = collect(0:0.1:1e3)
		minT1 = 1; maxT1 = 12

		AllData = SharedArrays.SharedArray{Float64,2}(maxT1 - minT1 + 2, length(tvec))
		T1vec = SharedArrays.SharedArray{Float64,2}(maxT1 - minT1 + 1, 2)
		AllData[1,:] = tvec
		runtime = @timed @sync @distributed for i in minT1:maxT1
			# ===== Fixed X estate evolution ======
			# fcop = [GamP_60[i],GamP_60[i],GamR_X[i],GamR_X[i]]
			# data = meEvolveState(initialMats, targetMats, HP + Ws_X[i]*HS + Ohm_X[i]*HX, cops, fcop, [], [], tvec, false)

			# ===== Fixed X estate evolution with Y params======
			fcop = [GamP_60[i],GamP_60[i],GamR_Y[i],GamR_Y[i]]
			data = meEvolveState(initialMats, targetMats, HP + Ws_Y[i]*HS + Ohm_Y[i]*HX, cops, fcop, [], [], tvec, false)

			T1vec[i, 1] = Int(1e-3/GamP_60[i])
			T1vec[i, 2] = 1e-3getLifetime(data, 2, 0, 0.99, 1)

			# Pulse-reset evolution
			# locParams = copy(pulseParams)
			# locParams[:, 2] = trVecTEST[i]*ones(2)
			# locParams[:,2] = tf*ones(2)
			# tr = locParams[1,2]
			# locFcop = zeros(4, length(tvec)); locFHam = zeros(2, length(tvec))
			# locFcop[1:2,:] = GamP_60[i]*ones(2, length(tvec)); # locFcop[3:4,:] = GamR[i]*ones(2, length(tvec))

			# for j in 1:length(tvec)
			# 	locFcop[3:4, j] = block(tvec[j], [tp, tr, GamR_0, GamP_60[i]])*ones(2)
			# 	locFHam[1, j] = pulse(tvec[j], locParams[1,:])
			# 	locFHam[2, j] = pulse(tvec[j], locParams[2,:])
			# end
			# data = meEvolveState(initialMats, targetMats, HP + Ws_0*HS, cops, locFcop, [HX, HY], locFHam, tvec, true)
			# data = meEvolveState(initialMats, targetMats, HP + Ws_X[i]*HS, cops, locFcop, [HX, HY], locFHam, tvec, true)
			# truncData = copy(data[:, 1])
			# for j in 1:length(tvec)
			# 	n = Int(floor(tvec[j]/(tp + tr)))
			# 	if (tvec[j] == ((n + 1)*tp + n*tr))
			# 		truncData = hcat(truncData, data[:, j])
			# 	end
			# end
			# T1vec[i, 1] = Int(1e-3/GamP_60[i])
			# T1vec[i, 2] = 1e-3getLifetime(truncData, 2, 0, 0.99, 1)

			AllData[i-minT1+2,:] = data[2,:]
		end
		file1 = open(datafile, "w")
		writedlm(file1, AllData)
		close(file1)
		file2 = open(lifetimesfile, "w")
		writedlm(file2, T1vec)
		close(file2)
		printRuntime(runtime[2])
	end

	function scan(T1index)
		initialMats = [rhoL0]
		targetMats = [pLX]
		tp = 40
		# pulseRaw = readdlm("output/vslq-tp=$(tp)-W=$(W/(2pi))-PULSEDE.pls")
		# pulseX = hcat([tp 0], pulseRaw[1:20]')
		# pulseY = hcat([tp 0], pulseRaw[21:end]')
		# pulseParams = vcat(pulseX, pulseY)
		pulseParams = readdlm("output/vslq-tp=$(tp)-W=$(W/(2pi))-PULSEGRAPE.pls")
		tvec = collect(0:0.05:2e3)
		filename = "output/X-tp=$(tp)-W=$(W/(2pi))-T1=$(5T1index)-SCAN.dat"
        mintr = 30; maxtr = 300
		fcop = zeros(4, length(tvec))
		fcop[1:2,:] = GamP_60[T1index]*ones(2, length(tvec))
		fHam = zeros(2, length(tvec))
		T1vec = SharedArrays.SharedArray{Float64,2}(maxtr - mintr, 2)
		println("Entering SCAN with $(nprocs()) processors")
		runtime = @timed @sync @distributed for tr in (mintr + 1):maxtr
			locParams = copy(pulseParams)
			locParams[:, 2] = tr*ones(2)
			for j in 1:length(tvec)
				fcop[3:4, j] = block(tvec[j], [tp, tr, GamR_0, GamP_60[T1index]])*ones(2)
				fHam[1, j] = pulse(tvec[j], locParams[1,:])
				fHam[2, j] = pulse(tvec[j], locParams[2,:])
			end
			data = meEvolveState(initialMats, targetMats, HP + Ws_0*HS, cops, fcop, [HX, HY], fHam, tvec, true)
			truncData = copy(data[:, 1])
			for j in 1:length(tvec)
				n = Int(floor(tvec[j]/(tp + tr)))
				if (tvec[j] == ((n + 1)*tp + n*tr))
					truncData = hcat(truncData, data[:, j])
				end
			end

			T1vec[tr - mintr, 2] = tr
			T1vec[tr - mintr, 3] = 1e-3getLifetime(truncData, 2, 0, 0.99, 1)
		end
		println("Output to ", filename)
		outfile = open(filename, "w")
		writedlm(outfile, T1vec)
		write(outfile, "\nBest result :: $(T1index), $(T1vec[findmax(T1vec[:,2])[2],:])\n")
		close(outfile)
		printRuntime(runtime[2])
	end
	
	function pulsedGrape()
		tp = 40
		println("PULSE GRAPE")
		initialStates = [psiERr1, psiL0]
		targetStates = [psiL0c, psiL0]
		paramsHam = [                                     
			tp tf 0.055;
			tp tf 0
		]
		tvec = collect(0:0.05:tp)
		filename = "vslq-tp=$(tp)-W=$(W/(2pi))"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed pulseGRAPE(filename, initialStates, H, Ht, targetStates, tvec, paramsHam, [2, 5], 1e-4, 1e-6, 20, 10000)
		println("Output to ", filename, "-PULSEGRAPE")
		printRuntime(runtime[2])
	end

	function pulseOptim()
		tp = 60
		println("PULSE DIFFERENTIAL EVOLUTION")
		initialStates = [psiERr1, psiL0]
		targetStates = [psiL0c, psiL0]
		paramsHam = [                                     
			tp tf 0.04;
			tp tf 0
		]
		tvec = collect(0:0.05:tp)
		filename = "vslq-tp=$(tp)-W=$(W/(2pi))"
		# println("Entering Differential Evolution with $(nprocs()) processors")
		# runtime = @timed pulseDiffEvo(filename, initialStates, H, Ht, targetStates, tvec, paramsHam, [2, 5], 0.03, 0.01, 20, 50000)
		rangeBig = 0.02
		rangeSmall = 0.001
		searchSpace = [
			(paramsHam[1, 3] - rangeBig, paramsHam[1, 3] + rangeBig), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall),
			(paramsHam[2, 3] - rangeSmall, paramsHam[2, 3] + rangeSmall), (-rangeBig, rangeBig), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall), (-rangeSmall, rangeSmall)
		]
		runtime = @timed pulseDiffEvo(filename, initialStates, H, Ht, targetStates, tvec, paramsHam, [2, 5], 20, searchSpace, 100000)
		println("Output to ", filename, "-PULSEDE")
		printRuntime(runtime[2])
	end

	function fixedGrape(T1index)
		println("FIXED GRAPE $(5T1index)")
		initialMats = [rhoL0]
		targetMats = [pLX]
		tvec = collect(0:0.1:2e3)
		maxit = 1.5e3
		filename = "X60-$(5T1index)"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed fixedGRAPE(filename, initialMats, HP, [HS, HX], targetMats, tvec, [[Ws_0, Ohm_0],[GamR_0,GamR_0]], 2, 1e-8, 0, cops, [GamP_60[T1index],GamP_60[T1index]], maxit)
		println("Output to ", filename, "-FIXEDGRAPE")
		printRuntime(runtime[2])
	end

	function fixedOptim(T1index)
		println("FIXED DIFFERENTIAL EVOLUTION $(5T1index)")
		initialMats = [rhoL0]
		targetMats = [pLX]
		tvec = collect(0:0.1:2e3)
		maxit = 10000
		filename = "vslq-W=$(W/(2pi))-T1=$(5T1index)"
		# println("Entering GRAPE with $(nprocs()) processors")
		searchSpace = [(Ws_0 - 0.002, Ws_0 + 0.002), (0.005, 0.04), (0.005, 0.035), (0.005, 0.035)]
		runtime = @timed fixedDiffEvo(filename, initialMats, HP, [HS, HX], targetMats, tvec, [[Ws_0, Ohm_0],[GamR_0,GamR_0]], 2, 0, cops, [GamP_60[T1index],GamP_60[T1index]], searchSpace, maxit)
		println("Output to ", filename, "-FIXEDDE")
		printRuntime(runtime[2])
	end

	@match action begin
		"regular" => regularEvolution()
		# "pulse" => pulsedGrape()
		"pulse" => pulseOptim()
		"fixed" => fixedOptim(T1index)
		# "fixed" => fixedGrape(T1index)
		"scan" => scan(T1index)
		_ => println("Invalid option")
	end

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

function threeQubit(action, timedep, T1index)
	println("THREE QUBIT")
	J = 2pi*0.02
	k0::Array{Complex{Float64},1} = [1,0]; k1::Array{Complex{Float64},1} = [0,1];
	r0 = k0*k0'; r1 = k1*k1'
	a = [0 1; 0 0]
	X = [0 1; 1 0]; Y = [0 -1; 1 0]im; Z = [1 0; 0 -1]
	eye = [1 0; 0 1]
	bigEye = tensor(eye,eye,eye)

	ket000 = tensor(k0,k0,k0,k0,k0,k0); rho000 = sparse(ket000*ket000')
    ket111 = tensor(k1,k1,k1,k0,k0,k0); rho111 = sparse(ket111*ket111')
    ket100 = tensor(k1,k0,k0,k0,k0,k0); rho100 = sparse(ket100*ket100')
    # ket010 = tensor(k0,k1,k0,k0,k0,k0); rho010 = sparse(ket010*ket010')
    # ket001 = tensor(k0,k0,k1,k0,k0,k0); rho001 = sparse(ket001*ket001')
    # ket110 = tensor(k1,k1,k0,k0,k0,k0); rho110 = sparse(ket110*ket110')
    # ket101 = tensor(k1,k0,k1,k0,k0,k0); rho101 = sparse(ket101*ket101')
    # ket011 = tensor(k0,k1,k1,k0,k0,k0); rho011 = sparse(ket011*ket011')
	k000100 = tensor(k0,k0,k0,k1,k0,k0);
	rhoL1 = sparse(tensor(r1,r1,r1,bigEye))
	rhoL0 = sparse(tensor(r0,r0,r0,bigEye))
	majL1 = rhoL1 + sparse(tensor(r1,r1,r0,bigEye) + tensor(r0,r1,r1,bigEye) + tensor(r1,r0,r1,bigEye))
	majL0 = rhoL0 + sparse(tensor(r0,r0,r1,bigEye) + tensor(r1,r0,r0,bigEye) + tensor(r0,r1,r0,bigEye))

	X1 = sparse(tensor(X,eye,eye,eye,eye,eye)); X1R = sparse(tensor(eye,eye,eye,X,eye,eye));
	Y1 = sparse(tensor(Y,eye,eye,eye,eye,eye)); Y1R = sparse(tensor(eye,eye,eye,Y,eye,eye));
	Z1 = sparse(tensor(Z,eye,eye,eye,eye,eye)); Z1R = sparse(tensor(eye,eye,eye,Z,eye,eye));
	X2 = sparse(tensor(eye,X,eye,eye,eye,eye)); X2R = sparse(tensor(eye,eye,eye,eye,X,eye));
	Y2 = sparse(tensor(eye,Y,eye,eye,eye,eye)); Y2R = sparse(tensor(eye,eye,eye,eye,Y,eye));
	Z2 = sparse(tensor(eye,Z,eye,eye,eye,eye)); Z2R = sparse(tensor(eye,eye,eye,eye,Z,eye));
	X3 = sparse(tensor(eye,eye,X,eye,eye,eye)); X3R = sparse(tensor(eye,eye,eye,eye,eye,X));
	Y3 = sparse(tensor(eye,eye,Y,eye,eye,eye)); Y3R = sparse(tensor(eye,eye,eye,eye,eye,Y));
	Z3 = sparse(tensor(eye,eye,Z,eye,eye,eye)); Z3R = sparse(tensor(eye,eye,eye,eye,eye,Z));
	a1 = sparse(tensor(eye,eye,eye,a,eye,eye)); a1d = a1'
	a2 = sparse(tensor(eye,eye,eye,eye,a,eye)); a2d = a2'
	a3 = sparse(tensor(eye,eye,eye,eye,eye,a)); a3d = a3'

	EYE = sparse(tensor(bigEye, bigEye))

	P111 = 0.125(EYE - Z1)*(EYE - Z2)*(EYE - Z3)

	# println(P111 == rhoL1)

	HPZ = Z1*Z2 + Z2*Z3 + Z1*Z3
	HRZ = Z1R + Z2R + Z3R
	HP = -J*(Z1*Z2 + Z2*Z3 + Z1*Z3)
	HR = -2J*(Z1R + Z2R + Z3R)
	HX = X1*X1R + X2*X2R + X3*X3R
	HY = X1*Y1R + X2*Y2R + X3*Y3R
	
	H = HP + HR
	cops = [X1,X2,X3,a1,a2,a3]
	Ht = [HX, HY]

	wR_0 = -2J
	Ohm_0 = 0.015
	GamR_0 = 0.027

	tp = 40; tf = 100
	# GamP = 1e-3./(10*[1,2,3,4,5,6,7,8,9,10])
	GamP = 1e-3collect(10:10:100)
	# GamR = 0.03
	wR = [-0.250784, -0.250983, -0.25107, -0.251123, -0.251159, -0.251139, -0.251162, -0.25118, -0.251196, -0.251209]
	Ohm = [0.0208399, 0.0179989, 0.0165715, 0.0156172, 0.014904, 0.0148725, 0.0143337, 0.0138823, 0.0134954, 0.013158]
	GamR = [0.030402, 0.0284754, 0.0281406, 0.0280666, 0.0280665, 0.0263955, 0.0262599, 0.0261655, 0.0260978, 0.0260481]
	trVec = [81, 93, 97, 122, 131, 143, 152, 152, 185, 197]

	function regularEvolution()
		initialMats = [rho111]
		targetMats = [rhoL1]
		# filename = "output/threeQubit-FIXED-EVOLUTION-ALL.dat"
		# filename = "output/threeQubit-TR-FLUSHED-EVOLUTION-ALL.dat"
		filename = "output/new.dat"
		pulseParams = readdlm("output/threeQubit-tp=40-J=0.02-PULSEGRAPE.pls"); #pulseParams[:, 2] = zeros(2)

		fileold = "output/old.dat"
		oldParams = readdlm("output/pls_40_1_XY.dat")

		tvec = collect(0:0.1:40)
		minT1 = 10; maxT1 = 10
		# tvec = collect(0:0.1:1e3)

		OldData = SharedArrays.SharedArray{Float64,2}(maxT1 - minT1 + 2, length(tvec))
		OldData[1,:] = tvec

		AllData = SharedArrays.SharedArray{Float64,2}(maxT1 - minT1 + 2, length(tvec))
		AllData[1,:] = tvec
		runtime = @timed @sync @distributed for i in minT1:maxT1
			# fcop = [GamP[i],GamP[i],GamP[i],GamR[i],GamR[i],GamR[i]]
			# data = meEvolveState(initialMats, targetMats, HP + wR[i]*HRZ + Ohm[i]*HX, cops, fcop, [], [], tvec, false)
			locParams = copy(pulseParams)
			locParams[:, 2] = trVec[i]*ones(2)
			tp = locParams[1,1]; tr = locParams[1,2]
			locFcop = zeros(6, length(tvec)); locFHam = zeros(2, length(tvec))
			locFcop[1:3,:] = GamP[i]*ones(3, length(tvec)); # locFcop[4:6,:] = GamR[i]*ones(3, length(tvec))

			locOld = copy(oldParams)
			locOld[:, 2] = trVec[i]*ones(2)			
			oldFcop = zeros(6, length(tvec)); oldFHam = zeros(2, length(tvec))
			oldFcop[1:3,:] = GamP[i]*ones(3, length(tvec))

			for j in 1:length(tvec)
				locFcop[4:6, j] = block(tvec[j], [tp, tr, GamR[1], GamP[i]])*ones(3)
				locFHam[1, j] = pulse(tvec[j], locParams[1,:])
				locFHam[2, j] = pulse(tvec[j], locParams[2,:])
				oldFcop[4:6, j] = block(tvec[j], [tp, tr, GamR[1], GamP[i]])*ones(3)
				oldFHam[1, j] = pulse(tvec[j], locOld[1,:])
				oldFHam[2, j] = pulse(tvec[j], locOld[2,:])
			end
			data = meEvolveState(initialMats, targetMats, HP + wR[i]*HRZ, cops, locFcop, [HX, HY], locFHam, tvec, true)
			AllData[2,:] = data[2,:]
			# AllData[i+1,:] = data[2,:]
			dataOld = meEvolveState(initialMats, targetMats, HP + wR[i]*HRZ, cops, oldFcop, [HX, HY], oldFHam, tvec, true)
			OldData[2,:] = dataOld[2,:]
			# OldData[i+1,:] = dataOld[2,:]
		end
		file = open(filename, "w")
		old = open(fileold, "w")
		writedlm(file, AllData)
		close(file)
		writedlm(old, OldData)
		close(old)
		printRuntime(runtime[2])
	end

	function scan(T1index)
		initialMats = [rho111]
		targetMats = [rhoL1]
		# i = 1
		filename = "output/threeQubit-T1=$(10T1index)-TR_SCAN.dat"
		# filename = "test.dat"
		pulseParams = readdlm("output/threeQubit-tp=40-J=0.02-PULSEGRAPE.pls")
		tvec = collect(0:0.05:1e3)
		tp = pulseParams[1,1];
        mintr = 50; maxtr = 250
		fcop = zeros(6, length(tvec))
		fcop[1:3,:] = GamP[T1index]*ones(3, length(tvec))
		fHam = zeros(2, length(tvec))
		T1vec = SharedArrays.SharedArray{Float64,2}(maxtr - mintr, 2)
		println("Entering SCAN with $(nprocs()) processors")
		runtime = @timed @sync @distributed for tr in (mintr + 1):maxtr
			locParams = copy(pulseParams)
			locParams[:, 2] = tr*ones(2)
			for j in 1:length(tvec)
				fcop[4:6, j] = block(tvec[j], [tp, tr, GamR[1], GamP[T1index]])*ones(3)
				fHam[1, j] = pulse(tvec[j], locParams[1,:])
				fHam[2, j] = pulse(tvec[j], locParams[2,:])
			end
			data = meEvolveState(initialMats, targetMats, HP + wR[T1index]*HRZ, cops, fcop, [HX, HY], fHam, tvec, true)
			truncData = copy(data[:, 1])
			for j in 1:length(tvec)
				n = Int(floor(tvec[j]/(tp + tr)))
				if (tvec[j] == ((n + 1)*tp + n*tr))
					truncData = hcat(truncData, data[:, j])
				end
			end

			T1vec[tr - mintr, 1] = tr
			T1vec[tr - mintr, 2] = 1e-3getLifetime(truncData, 2, 0.5, 0.9, 1)
		end
		println("Output to ", filename)
		file = open(filename, "w")
		writedlm(file, T1vec)
		close(file)
		printRuntime(runtime[2])
	end
	
	function pulsedGrape()
		initialStates = [ket100, ket000, ket111]
		targetStates = [k000100, ket000, ket111]
		paramsHam = [
			tp tf 0.1;
			tp tf 0
		]
        # paramsHam = readdlm("output/pls_40_1_XY.dat")
		# paramsHam = readdlm("output/threeQubit-tp=40-J=0.02-PULSEGRAPE.pls")
		tvec = collect(0:0.05:tp)
		filename = "threeQubit-tp=$(tp)-J=$(J/(2pi))"
		# filename = "OLD"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed pulseGRAPE(filename, initialStates, H, Ht, targetStates, tvec, paramsHam, [2,6,10], 1e-4, 1e-6, 20, 10000)
		println("Output to ", filename, "-PULSEGRAPE")
		printRuntime(runtime[2])
	end

	function fixedGrape(T1index)
		initialMats = [rho111]
		targetMats = [rhoL1]
		tvec = collect(0:0.5:1e3)
		i = 6
        maxit = 2000
        filename = "threeQubit-T1=$(10T1index)-NEW"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed fixedGRAPE(filename, initialMats, HP, [HRZ, HX], targetMats, tvec, [[wR_0, Ohm_0],[GamR_0,GamR_0,GamR_0]], 2, 1e-8, 0, cops, [GamP[T1index],GamP[T1index],GamP[T1index]], maxit)
		println("Output to ", filename, "-FIXEDGRAPE")
		printRuntime(runtime[2])
	end

	@match action begin
		"regular" => regularEvolution()
		"pulse" => pulsedGrape()
		"fixed" => fixedGrape(T1index)
		"scan" => scan(T1index)
		_ => println("Invalid option")
	end
end

function singleQubit(action, timedep, T1index)
	println("SINGLE QUBIT")
	# d = 2pi*0.35
	d = 2pi*0.2
	# d = 2pi*0.1
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

	psi00 = tensor(k0, ks0); rho00 = psi00*psi00'
	psi01 = tensor(k0, ks1); rho01 = psi01*psi01'
	psi10 = tensor(k1, ks0); rho10 = psi10*psi10'
	psi11 = tensor(k1, ks1); rho11 = psi11*psi11'
	psi20 = tensor(k2, ks0); rho20 = psi20*psi20'
	psi21 = tensor(k2, ks1); rho21 = psi21*psi21'

	initialStates = [psi00, psi10]
	targetStates = [psi00, psi01, psi10, psi11, psi20, psi21]
	rhoList = [rho00, rho01, rho10, rho11, rho20, rho21]
	keyRho = [rho00, rho10, rho11]
	initialMats = [rho10]
	targetMats = [rho10]
	cops = [aq, ar]
	Ht = [HX, HY]

	tp = 20; tr = 400
	GamQ = 0.0001
	GamP = 1e-3./(10*[1,2,3,4,5,6,7,8,9,10])
	GamR = 0.03
	Ohm = 0.005
	plsParams = readdlm("pls_20.dat"); plsParams[:,2:end] *= 2pi
	# plsParams = [
	# 	tp tr 0.12;
	# 	tp tr 0
	# ]

	function getResidual(timedep, T1index)
		tvec = collect(0:0.05:tp+tr)
		matrixM = zeros(6,6)
		tiFcop = [GamP[T1index], GamR]
		locParams = hcat([tr; tr], plsParams)
		fcop = zeros(2, length(tvec))
		fHam = zeros(2, length(tvec))
		fcop[1,:] = GamP[T1index]*ones(1, length(tvec))
		for i in 1:length(tvec)
			fcop[2, i] = block(tvec[i], [tr, tp, GamR, GamP[T1index]])
			fHam[1, i] = pulse(tvec[i], locParams[1,:])
			fHam[2, i] = pulse(tvec[i], locParams[2,:])
		end
		# data = meEvolveState(rhoList, rhoList, HP + Ohm*HX, cops, fcop, [], [], tvec, false)
		# data = meEvolveState(rhoList, rhoList, HP, cops, fcop, [HX, HY], fHam, tvec, true)
		data = meEvolveState(rhoList, rhoList, timedep ? HP : HP + Ohm*HX, cops, timedep ? fcop : tiFcop, timedep ? [HX, HY] : [], timedep ? fHam : [], tvec, timedep)
		bestTrIndex = findmax(data[4,:])[2]
		bestTr = data[1, bestTrIndex]
		for i in 1:6
			for j in 1:6
				matrixM[j, i] = data[(i-1)*6 + j + 1, bestTrIndex]
			end
		end
		esys = eigen(matrixM)
		prob00to10 = (esys.vectors[:,end]/sum(esys.vectors[:,end]))[3]

		fout = open("test", "w")
		writedlm(fout, data)
		close(fout)
		display(matrixM); println(); println()
		display(esys); println(); println()
		println("P(|00> -> |10>) :: ", prob00to10)
		# return data
	end

	function optResidual()

	end

	function regularEvolution()
		# filename = "singleQubit-EVOLUTION.dat"
		tvec = collect(0:0.05:5000tp)
		fcop = [1e-3/10, GamR]
		rt1 = @timed data = meEvolveState(initialMats, targetMats, HP + Ohm*HX, cops, fcop, [], [], tvec, false)
		# file = open(filename, "w")
		# writedlm(file, data)
		# close(file)
		rt2 = @timed for i in 2:5
			fcop = [1e-3/(10i), GamR]
			data = meEvolveState(initialMats, targetMats, HP + Ohm*HX, cops, fcop, [], [], tvec, false)
			file = open(filename, "a")
			writedlm(file, data[2,:]')
			close(file)
		end
		runtime = rt1[2] + rt2[2]
		printRuntime(runtime)
	end
	
	function pulsedGrape()
		tvec = collect(0:0.05:tp)
		# filename = "singleQubit-tp=$(tp)-d=$(d/(2pi))"
		filename = "TEST"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed pulseGRAPE(filename, initialStates, HP, Ht, targetStates, tvec, paramsHam, [5, 10], 1e-4, 1e-6, 20, 5000)
		println("Output to ", filename, "-PULSEGRAPE")
		printRuntime(runtime[2])
	end

	function fixedGrape(T1index)
		tvec = collect(0:0.05:10000)
		# i = 10
        maxit = 1
		filename = "singleQubit-T1=$(10T1index)"
		println("Entering GRAPE with $(nprocs()) processors")
		runtime = @timed fixedGRAPE(filename, initialMats, HP, [HX], targetMats, tvec, [[Ohm],[GamR]], 2, 1e-4, 0, cops, [GamP[T1index]], maxit)
		# runtime = @timed fixedGRAPE(filename, initialMats, 0HP, [-pq2, HX], targetMats, tvec, [[d, Ohm],[GamR]], 2, 1e-4, 0, cops, [GamP[i]], maxit)
		println("Output to ", filename, "-FIXEDGRAPE")
		printRuntime(runtime[2])
	end

	function fixedOptim(T1index)
		tvec = collect(0:0.05:10000)
        maxit = 1
		filename = "singleQubit-T1=$(10T1index)"
		# runtime = @timed fixedDiffEvo(filename, initialState, H0, Ht, target, tvec, params, index, correction, collapseOps, collapsePrimary, searchSpace, maxit)
		# runtime = @timed fixedGRAPE(filename, initialMats, 0HP, [-pq2, HX], targetMats, tvec, [[d, Ohm],[GamR]], 2, 1e-4, 0, cops, [GamP[i]], maxit)
		println("Output to ", filename, "-FIXEDGRAPE")
		printRuntime(runtime[2])
	end

	if action == "regular"
		regularEvolution()
	elseif action == "pulse"
		pulsedGrape()
	elseif action == "fixed"
		fixedGrape(T1index)
	elseif action == "residual"
		getResidual(timedep, T1index)
	else
		println("Invalid Option")
	end
end

function main(system, action, tdep="false", T1="1")
	timedep = tdep .== "true"
	T1index = parse(Int, T1)
	if system == "singleQubit"
		singleQubit(action, timedep, T1index)
	elseif system == "threeQubit"
		threeQubit(action, timedep, T1index)
	elseif system == "vslq"
		vslq(action, timedep, T1index)
	else
		println("Invalid Option")
	end
end
main(args::AbstractVector) = main(args...)

main(ARGS)