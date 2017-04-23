

require 'nn'

-- network-----mini size net for cpu test runs only--------------------------------------------------
function getModel()
	nbFilter=1
	input = nn.Identity()()
	conv1 = nn.SpatialConvolution(3, nbFilter, 3, 3)(input)
	View_1=nn.View(1198198)(conv1)
	out=nn.ReLU()(nn.Linear(1198198, 3)(View_1))
	gmod = nn.gModule({input}, {out})
	local method = 'xavier'
	local gmod = require('weight-init')(gmod, method)
	return gmod
end
