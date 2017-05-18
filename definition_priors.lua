function get_Rep_criterion()
   h1 = nn.Identity()()
   h2 = nn.Identity()()
   h3 = nn.Identity()()
   h4 = nn.Identity()()

   h_h1 = nn.CSubTable()({h2,h1})
   h_h2 = nn.CSubTable()({h4,h3})

   madd = nn.CSubTable()({h_h2,h_h1})
   sqr=nn.Square()(madd)
   out1 = nn.Sum(1,1)(sqr)

   norm2= nn.Sum(1,1)(nn.Square()(nn.CSubTable()({h3,h1})))
   out2=nn.Exp()(nn.MulConstant(-1)(norm2))

   outTot=nn.Sum(1,1)(nn.CMulTable()({out1, out2}))
   gmod = nn.gModule({h1, h2, h3, h4}, {outTot})
   return gmod
end

function get_Prop_criterion()
   h1 = nn.Identity()()
   h2 = nn.Identity()()
   h3 = nn.Identity()()
   h4 = nn.Identity()()

   h_h1 = nn.CSubTable()({h2,h1})
   h_h2 = nn.CSubTable()({h4,h3})

   norm=nn.Sqrt()(nn.Sum(1,1)(nn.Square()(h_h1)))
   norm2=nn.Sqrt()(nn.Sum(1,1)(nn.Square()(h_h2)))

   madd = nn.CSubTable()({norm,norm2})
   sqr=nn.Square()(madd)
   out = nn.Sum(1,1)(sqr)

   gmod = nn.gModule({h1, h2, h3, h4}, {out})
   return gmod
end

function get_Caus_criterion()
   h1 = nn.Identity()()
   h2 = nn.Identity()()

   h_h1 = nn.CSubTable()({h2,h1})

   norm=nn.Sum(1,1)(nn.Square()(h_h1))
   exp=nn.Exp()(nn.MulConstant(-1)(norm))
   out = nn.Sum(1,1)(exp)

   gmod = nn.gModule({h1, h2}, {out})
   return gmod
end
