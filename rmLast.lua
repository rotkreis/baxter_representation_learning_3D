-- Handy script to delete last model used
require 'functions'

if file_exists('lastModel.txt') then
   f = io.open('lastModel.txt','r')
   path = f:read()
   modelString = f:read()
   print('MODEL USED : '..modelString)
   f:close()
else
   error("lastModel.txt should exist")
end

print("Do you really want to delete last model ? Enter if okay Ctrl-C otherwise")
io.read()

os.execute("rm -r "..path)
print("Deleted last model successfully")
